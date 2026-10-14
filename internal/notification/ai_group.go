package notification

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/amir20/dozzle/internal/ai"
	"github.com/amir20/dozzle/internal/container"
	"github.com/amir20/dozzle/internal/utils"
	"github.com/amir20/dozzle/types"
	"github.com/rs/zerolog/log"
)

// Explainer explains a group of matched log lines. *ai.Explainer is the real one.
type Explainer interface {
	Explain(ctx context.Context, req ai.Request) ai.Result
}

const (
	// aiBudget bounds the whole AI step: the context read and both providers.
	aiBudget = 45 * time.Second
	// aiContextWindow is how far around the first match context lines are read,
	// and aiContextTimeout how long that may take: a read from a timestamp makes
	// Docker decode the log file from its start, which must not eat the budget.
	aiContextWindow  = 30 * time.Second
	aiContextTimeout = 10 * time.Second
	// aiMaxContext bounds the lines held on each side while context is read.
	aiMaxContext = 1000
	// aiMaxDistinct bounds the distinct lines a group holds. Later ones still count.
	aiMaxDistinct = 100
	// aiMaxLine is the most of one matched line that a model or an alert shows.
	aiMaxLine = 2048
	// maxDetail is Slack's limit for one section.
	maxDetail = 3000
)

type aiGroupKey struct {
	subID       int
	containerID string
}

// aiBatch is what one grouped notification reports.
type aiBatch struct {
	container types.NotificationContainer // as it was at the first match
	first     types.NotificationLog
	firstAt   time.Time
	lastAt    time.Time
	matches   []ai.Match // distinct, in the order first seen
	index     map[string]int
	count     int
}

// SetExplainer turns on AI explanations for the rules that ask for them. Call it before Start.
func (m *Manager) SetExplainer(e Explainer) {
	m.explainer = e
}

// groupsForAI reports whether sub's matches wait to be grouped and explained.
// Cloud does its own triage on the raw line, so its rules keep sending each match.
func (m *Manager) groupsForAI(sub *Subscription) bool {
	return m.explainer != nil && sub.AISummary && sub.DispatcherID != 0
}

// recordAIMatch adds a match to its rule's group on that container. The first
// match starts a short settle, so a burst goes out as one notification.
func (m *Manager) recordAIMatch(subID int, c types.NotificationContainer, l types.NotificationLog, e *container.LogEvent) {
	text, stack := aiText(e)
	at := time.UnixMilli(l.Timestamp)
	key := aiGroupKey{subID, c.ID}

	m.aiMu.Lock()
	defer m.aiMu.Unlock()
	b, ok := m.aiGroups[key]
	if !ok {
		b = &aiBatch{}
		m.aiGroups[key] = b
		time.AfterFunc(m.aiSettle, func() { m.flushAIGroup(key) })
	}
	if b.count == 0 {
		*b = aiBatch{container: c, first: l, firstAt: at, index: make(map[string]int)}
	}
	b.count++
	b.lastAt = at
	shape := ai.Normalize(text)
	if i, ok := b.index[shape]; ok {
		b.matches[i].Count++
	} else if len(b.matches) < aiMaxDistinct {
		b.index[shape] = len(b.matches)
		b.matches = append(b.matches, ai.Match{Text: ai.Truncate(text, aiMaxLine), Offset: at.Sub(b.firstAt), Stack: stack, Count: 1})
	}
}

// flushAIGroup sends what a group collected and starts its quiet period, during
// which matches wait for the next flush. A quiet period that ends with nothing
// new closes the group, so the map can't grow without bound.
func (m *Manager) flushAIGroup(key aiGroupKey) {
	m.aiMu.Lock()
	b, ok := m.aiGroups[key]
	if !ok || b.count == 0 {
		delete(m.aiGroups, key)
		m.aiMu.Unlock()
		return
	}
	batch := *b
	*b = aiBatch{}
	time.AfterFunc(m.aiQuiet, func() { m.flushAIGroup(key) })
	m.aiMu.Unlock()

	m.sendAIGroup(key.subID, batch)
}

// sendAIGroup explains a batch and sends it as one notification. The rule is
// looked up again, because replace, update and config sync swap the pointer.
func (m *Manager) sendAIGroup(subID int, b aiBatch) {
	sub, ok := m.subscriptions.Load(subID)
	if !ok || !sub.Enabled {
		log.Debug().Int("subscription", subID).Msg("Dropped a grouped alert: its rule is gone or disabled")
		return
	}
	d, ok := m.getDispatcher(sub.DispatcherID)
	if !ok {
		return
	}

	// A rule unticked or moved to Cloud while the group settled goes out plain.
	var res ai.Result
	if m.groupsForAI(sub) {
		ctx, cancel := context.WithTimeout(m.ctx, aiBudget)
		res = m.explainer.Explain(ctx, ai.Request{
			Container: b.container.Name,
			Image:     b.container.Image,
			Host:      b.container.HostName,
			Matched:   b.matches,
			Count:     b.count,
			Span:      b.lastAt.Sub(b.firstAt),
			Context:   func(ctx context.Context) ([]ai.Line, []ai.Line) { return m.aiContext(ctx, b) },
		})
		cancel()
	}

	m.sendNotification(d, types.Notification{
		ID:           fmt.Sprintf("%s-%d", b.container.ID, time.Now().UnixNano()),
		Type:         types.LogNotification,
		Detail:       aiDetail(res, b),
		Container:    b.container,
		Log:          &b.first,
		Subscription: logSubscriptionConfig(sub),
		Timestamp:    time.Now(),
	}, sub.DispatcherID)
}

// aiContext reads the lines around a batch's first match, from aiContextWindow
// before it until as long after it, or now.
func (m *Manager) aiContext(ctx context.Context, b aiBatch) (before, after []ai.Line) {
	c, client, err := m.listener.findContainerAndClient(ctx, b.container.ID, nil)
	if err != nil {
		log.Debug().Err(err).Str("containerID", b.container.ID).Msg("No context lines for the AI explanation")
		return nil, nil
	}
	to := b.firstAt.Add(aiContextWindow)
	if now := time.Now(); now.Before(to) {
		to = now
	}
	ctx, cancel := context.WithTimeout(ctx, aiContextTimeout)
	defer cancel() // also stops the reader when enough lines are in
	events, err := client.LogsBetweenDates(ctx, c, b.firstAt.Add(-aiContextWindow), to, container.STDALL)
	if err != nil {
		log.Debug().Err(err).Str("containerID", c.ID).Msg("No context lines for the AI explanation")
		return nil, nil
	}

	nearest := utils.NewRingBuffer[ai.Line](aiMaxContext)
	for e := range events {
		text, stack := aiText(e)
		line := ai.Line{Text: text, Offset: time.UnixMilli(e.Timestamp).Sub(b.firstAt), Stack: stack}
		if line.Offset < 0 {
			nearest.Push(line)
		} else if after = append(after, line); len(after) == aiMaxContext {
			break
		}
	}
	return nearest.Data(), after
}

// aiText is the text a model or an alert sees for one entry: JSON as the app
// wrote it, keys in order, and other lines without ANSI codes or the timestamp
// the app printed itself.
func aiText(e *container.LogEvent) (text string, stack bool) {
	switch m := e.Message.(type) {
	case string:
		return container.StripANSI(m[min(e.TimestampPrefix, len(m)):]), false
	case []container.LogFragment:
		lines := make([]string, len(m))
		for i, f := range m {
			lines[i] = container.StripANSI(f.Message[min(f.TimestampPrefix, len(f.Message)):])
		}
		return strings.Join(lines, "\n"), true
	}
	if e.RawMessage != "" {
		return e.RawMessage, false
	}
	return formatLogMessage(extractMessage(*e)), false
}

// aiDetail is a grouped notification's text: the explanation, then the most
// frequent distinct lines. The lines are redacted, since they reach chat
// channels, but keep their ids so they can still be searched for in Dozzle.
func aiDetail(res ai.Result, b aiBatch) string {
	var sb strings.Builder
	if res.Text != "" {
		sb.WriteString(res.Text)
		if !res.ReusedFrom.IsZero() {
			fmt.Fprintf(&sb, " (seen before, explained at %s)", res.ReusedFrom.Local().Format("15:04 MST"))
		}
		sb.WriteString("\n\n")
	}

	lines := slices.Clone(b.matches)
	slices.SortStableFunc(lines, func(x, y ai.Match) int { return y.Count - x.Count })
	if b.count > 1 {
		fmt.Fprintf(&sb, "%d matching lines, %d distinct:\n", b.count, len(lines))
	}
	for i, l := range lines[:min(3, len(lines))] {
		limit := 400
		if i == 0 {
			limit = 1200 // so a 5xx's cause and location survive
		}
		if l.Count > 1 {
			fmt.Fprintf(&sb, "×%d ", l.Count)
		}
		sb.WriteString(ai.Truncate(ai.Redact(l.Text), limit) + "\n")
	}
	if more := len(lines) - 3; more > 0 {
		fmt.Fprintf(&sb, "…and %d more\n", more)
	}
	return ai.Truncate(strings.TrimSpace(sb.String()), maxDetail)
}
