package notification

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/amir20/dozzle/internal/ai"
	"github.com/amir20/dozzle/internal/container"
	"github.com/amir20/dozzle/internal/notification/dispatcher"
	"github.com/amir20/dozzle/types"
	"github.com/puzpuzpuz/xsync/v4"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/sync/semaphore"
)

type fakeExplainer struct {
	mu       sync.Mutex
	requests []ai.Request
	result   ai.Result
}

func (f *fakeExplainer) Explain(_ context.Context, req ai.Request) ai.Result {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requests = append(f.requests, req)
	return f.result
}

func (f *fakeExplainer) calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.requests)
}

type chanDispatcher chan types.Notification

func (c chanDispatcher) Send(_ context.Context, n types.Notification) error {
	c <- n
	return nil
}

func newAITestManager(t *testing.T, explainer Explainer) (*Manager, chanDispatcher) {
	sent := make(chanDispatcher, 10)
	m := &Manager{
		subscriptions: xsync.NewMap[int, *Subscription](),
		dispatchers:   xsync.NewMap[int, dispatcher.Dispatcher](),
		ctx:           t.Context(),
		sendSem:       semaphore.NewWeighted(5),
		aiGroups:      make(map[aiGroupKey]*aiBatch),
		aiSettle:      20 * time.Millisecond,
		aiQuiet:       200 * time.Millisecond,
	}
	m.SetExplainer(explainer)
	m.dispatchers.Store(1, sent)
	m.subscriptions.Store(1, &Subscription{ID: 1, Name: "API 5xx", Enabled: true, DispatcherID: 1, AISummary: true})
	return m, sent
}

var apiContainer = types.NotificationContainer{ID: "c1", Name: "ternakklip_api", Image: "ternakklip/api:1.4.2", HostName: "prod"}

func recordMatch(m *Manager, text string, at time.Time) {
	l := types.NotificationLog{ID: uint32(at.UnixMilli()), Message: text, Timestamp: at.UnixMilli()}
	m.recordAIMatch(1, apiContainer, l, &container.LogEvent{Message: text, Timestamp: at.UnixMilli()})
}

func receive(t *testing.T, sent chanDispatcher) types.Notification {
	t.Helper()
	select {
	case n := <-sent:
		return n
	case <-time.After(2 * time.Second):
		t.Fatal("no notification was sent")
		return types.Notification{}
	}
}

func TestAIGroupSendsABurstOnce(t *testing.T) {
	explainer := &fakeExplainer{result: ai.Result{Text: "Cause: postgres is down. Check: the db container."}}
	m, sent := newAITestManager(t, explainer)

	start := time.Now()
	for i := range 5 {
		recordMatch(m, fmt.Sprintf(`{"status":500,"code":"DB_DOWN","req":"%d"}`, i), start.Add(time.Duration(i)*time.Second))
	}
	recordMatch(m, `{"status":503,"code":"UPSTREAM"}`, start.Add(6*time.Second))

	n := receive(t, sent)
	assert.Equal(t, types.LogNotification, n.Type)
	assert.Equal(t, apiContainer, n.Container)
	assert.Equal(t, `{"status":500,"code":"DB_DOWN","req":"0"}`, n.Log.Message, "the first match is the alert's log")
	assert.Equal(t, "API 5xx", n.Subscription.Name)
	assert.Equal(t, "Cause: postgres is down. Check: the db container.\n\n6 matching lines, 2 distinct:\n"+
		`×5 {"status":500,"code":"DB_DOWN","req":"0"}`+"\n"+`{"status":503,"code":"UPSTREAM"}`, n.Detail)

	require.Equal(t, 1, explainer.calls())
	req := explainer.requests[0]
	assert.Equal(t, "ternakklip_api", req.Container)
	assert.Equal(t, "ternakklip/api:1.4.2", req.Image)
	assert.Equal(t, 6, req.Count)
	assert.Equal(t, 6*time.Second, req.Span)
	assert.Equal(t, 6*time.Second, req.Matched[1].Offset)
	assert.Empty(t, sent, "one notification for the burst")
}

func TestAIGroupQuietPeriod(t *testing.T) {
	explainer := &fakeExplainer{}
	m, sent := newAITestManager(t, explainer)

	recordMatch(m, "boom", time.Now())
	first := receive(t, sent)
	assert.Equal(t, "boom", first.Detail)

	// During the quiet period matches wait, then go out together.
	recordMatch(m, "boom again", time.Now())
	recordMatch(m, "boom again", time.Now())
	select {
	case <-sent:
		t.Fatal("sent during the quiet period")
	case <-time.After(100 * time.Millisecond):
	}
	second := receive(t, sent)
	assert.Equal(t, "2 matching lines, 1 distinct:\n×2 boom again", second.Detail)
	assert.Equal(t, 2, explainer.calls())

	// A quiet period with nothing new closes the group.
	assert.Eventually(t, func() bool {
		m.aiMu.Lock()
		defer m.aiMu.Unlock()
		return len(m.aiGroups) == 0
	}, 2*time.Second, 10*time.Millisecond)
}

func TestAIGroupFollowsTheRuleAsItIsNow(t *testing.T) {
	tests := []struct {
		name     string
		change   func(m *Manager)
		wantSent bool
	}{
		{"deleted", func(m *Manager) { m.subscriptions.Delete(1) }, false},
		{"disabled", func(m *Manager) {
			m.subscriptions.Store(1, &Subscription{ID: 1, Enabled: false, DispatcherID: 1, AISummary: true})
		}, false},
		{"unticked", func(m *Manager) {
			m.subscriptions.Store(1, &Subscription{ID: 1, Enabled: true, DispatcherID: 1})
		}, true},
		{"moved to cloud", func(m *Manager) {
			cloud := make(chanDispatcher, 1)
			var d dispatcher.Dispatcher = cloud
			m.cloudDispatcher.Store(&d)
			m.subscriptions.Store(1, &Subscription{ID: 1, Enabled: true, DispatcherID: 0, AISummary: true})
			m.dispatchers.Store(0, cloud)
		}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			explainer := &fakeExplainer{result: ai.Result{Text: "Cause: x. Check: y."}}
			m, sent := newAITestManager(t, explainer)
			m.aiSettle = 100 * time.Millisecond
			recordMatch(m, "boom", time.Now())
			tt.change(m)

			if tt.wantSent {
				assert.Equal(t, "boom", receive(t, sent).Detail, "sent plain")
			} else {
				select {
				case n := <-sent:
					t.Fatalf("sent %q", n.Detail)
				case <-time.After(300 * time.Millisecond):
				}
			}
			assert.Zero(t, explainer.calls(), "no model call")
		})
	}
}

func TestGroupsForAI(t *testing.T) {
	m := &Manager{}
	ticked := &Subscription{AISummary: true, DispatcherID: 1}
	assert.False(t, m.groupsForAI(ticked), "no explainer means the old path, whatever the flag says")

	m.SetExplainer(&fakeExplainer{})
	assert.True(t, m.groupsForAI(ticked))
	assert.False(t, m.groupsForAI(&Subscription{DispatcherID: 1}))
	assert.False(t, m.groupsForAI(&Subscription{AISummary: true, DispatcherID: 0}), "Cloud rules are not grouped")
}

func TestAIText(t *testing.T) {
	plain := "2026/10/13 14:05:12 CRITICAL: payout 42 not recorded"
	text, stack := aiText(&container.LogEvent{Message: plain, TimestampPrefix: len("2026/10/13 14:05:12 ")})
	assert.Equal(t, "CRITICAL: payout 42 not recorded", text)
	assert.False(t, stack)

	gorm := "2026/10/13 09:45:00 \x1b[31;1m/app/repo.go:277 \x1b[35;1mduplicate key value"
	text, _ = aiText(&container.LogEvent{Message: gorm, TimestampPrefix: len("2026/10/13 09:45:00 ")})
	assert.Equal(t, "/app/repo.go:277 duplicate key value", text)

	gin := `[GIN] 2026/10/13 - 14:04:57 | 500 | 5.0s | 172.18.0.1 | GET "/x"`
	text, _ = aiText(&container.LogEvent{Message: gin})
	assert.Equal(t, gin, text, "a timestamp behind [GIN] is kept")

	raw := `{"msg":"500 X","status":500,"at":"a.go:1"}`
	fields := container.Fields[any]{}
	text, _ = aiText(&container.LogEvent{Type: container.LogTypeComplex, Message: fields, RawMessage: raw})
	assert.Equal(t, raw, text, "JSON goes out as written, msg first")

	text, stack = aiText(&container.LogEvent{Message: []container.LogFragment{
		{Message: "2026/10/13 14:10:03 runtime error: nil pointer", TimestampPrefix: len("2026/10/13 14:10:03 ")},
		{Message: "\x1b[31m/app/x.go:123\x1b[0m"},
	}})
	assert.Equal(t, "runtime error: nil pointer\n/app/x.go:123", text)
	assert.True(t, stack)
}

func TestAIDetail(t *testing.T) {
	cause := strings.Repeat("c", 1000)
	b := aiBatch{count: 9, matches: []ai.Match{
		{Text: "rare", Count: 1},
		{Text: `{"status":500,"cause":"` + cause + `","at":"repo.go:280"}`, Count: 5},
		{Text: "sent to budi@example.com, req 0aece9f2-d3f2-485f-9b48-697faa6bcd9a", Count: 2},
		{Text: "fourth", Count: 1},
	}}

	detail := aiDetail(ai.Result{}, b)
	assert.True(t, strings.HasPrefix(detail, "9 matching lines, 4 distinct:\n×5 {"), detail)
	assert.Contains(t, detail, `"at":"repo.go:280"}`, "the first line keeps up to 1,200 characters")
	assert.Contains(t, detail, "×2 sent to b***@example.com, req 0aece9f2-d3f2-485f-9b48-697faa6bcd9a", "redacted, ids kept")
	assert.Contains(t, detail, "\nrare\n…and 1 more")
	assert.NotContains(t, detail, "fourth")

	local := time.Local
	time.Local = time.FixedZone("WIB", 7*60*60)
	defer func() { time.Local = local }()
	reused := aiDetail(ai.Result{Text: "Cause: x. Check: y.", ReusedFrom: time.Date(2026, 10, 13, 7, 2, 0, 0, time.UTC)}, b)
	assert.True(t, strings.HasPrefix(reused, "Cause: x. Check: y. (seen before, explained at 14:02 WIB)\n\n9 matching lines"), reused)

	b.matches[1].Text = strings.Repeat("y", 2048)
	b.matches[2].Text = strings.Repeat("z", 2048)
	assert.LessOrEqual(t, len(aiDetail(ai.Result{Text: strings.Repeat("w", 600)}, b)), maxDetail)
}

// contextClient serves the lines around a match, as Docker would from its log file.
type contextClient struct {
	container.ClientService
	events []*container.LogEvent
}

func (c *contextClient) FindContainer(_ context.Context, id string, _ container.ContainerLabels) (container.Container, error) {
	return container.Container{ID: id}, nil
}

func (c *contextClient) LogsBetweenDates(_ context.Context, _ container.Container, from, to time.Time, _ container.StdType) (<-chan *container.LogEvent, error) {
	ch := make(chan *container.LogEvent, len(c.events))
	for _, e := range c.events {
		if ts := time.UnixMilli(e.Timestamp); !ts.Before(from) && !ts.After(to) {
			ch <- e
		}
	}
	close(ch)
	return ch, nil
}

func TestAIContext(t *testing.T) {
	first := time.Now().Add(-time.Minute).Truncate(time.Millisecond)
	client := &contextClient{events: []*container.LogEvent{
		{Message: "too early", Timestamp: first.Add(-time.Minute).UnixMilli()},
		{Message: "before", Timestamp: first.Add(-2 * time.Second).UnixMilli()},
		{Message: "the match", Timestamp: first.UnixMilli()},
		{Message: "after", Timestamp: first.Add(3 * time.Second).UnixMilli()},
		{Message: "too late", Timestamp: first.Add(time.Minute).UnixMilli()},
	}}
	m := &Manager{listener: NewContainerLogListener(t.Context(), []container.ClientService{client})}

	before, after := m.aiContext(t.Context(), aiBatch{container: apiContainer, firstAt: first})
	assert.Equal(t, []ai.Line{{Text: "before", Offset: -2 * time.Second}}, before)
	assert.Equal(t, []ai.Line{{Text: "the match"}, {Text: "after", Offset: 3 * time.Second}}, after)
}

// The flag has to survive every hop that rebuilds a rule: an edit through the
// rule list's toggle, then a save and a reload.
func TestAISummarySurvivesEditAndReload(t *testing.T) {
	newManager := func() *Manager {
		return NewManager(NewContainerLogListener(t.Context(), nil), NewContainerStatsListener(t.Context(), nil), NewContainerEventListener(t.Context(), nil))
	}
	m := newManager()
	sub := &Subscription{Name: "API 5xx", DispatcherID: 1, ContainerExpression: `name == "api"`, LogExpression: `message.status >= 500`, AISummary: true}
	require.NoError(t, m.AddSubscription(sub))
	require.NoError(t, m.UpdateSubscription(sub.ID, map[string]any{"enabled": false}))

	var buf bytes.Buffer
	require.NoError(t, m.WriteConfig(&buf))
	assert.Contains(t, buf.String(), "aiSummary: true")

	reloaded := newManager()
	require.NoError(t, reloaded.LoadConfig(&buf))
	require.Len(t, reloaded.Subscriptions(), 1)
	assert.True(t, reloaded.Subscriptions()[0].AISummary)
	assert.False(t, reloaded.Subscriptions()[0].Enabled)

	require.NoError(t, reloaded.UpdateSubscription(sub.ID, map[string]any{"aiSummary": false}))
	assert.False(t, reloaded.Subscriptions()[0].AISummary)
}
