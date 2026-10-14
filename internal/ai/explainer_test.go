package ai

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeProvider struct {
	calls atomic.Int32
	text  string
	err   error
	input string
}

func (f *fakeProvider) explain(_ context.Context, _, input string) (string, error) {
	f.calls.Add(1)
	f.input = input
	return f.text, f.err
}

func request(image, line string) Request {
	return Request{Container: "api", Image: image, Matched: []Match{match(line, 0, 1)}, Count: 1}
}

func TestExplainChain(t *testing.T) {
	tests := []struct {
		name            string
		claude, router  *fakeProvider
		want            string
		wantRouterCalls int32
	}{
		{"claude answers", &fakeProvider{text: "Cause: a. Check: b."}, &fakeProvider{}, "Cause: a. Check: b.", 0},
		{"claude fails, openrouter answers", &fakeProvider{err: errors.New("402")}, &fakeProvider{text: "Cause: c. Check: d."}, "Cause: c. Check: d.", 1},
		{"claude refuses", &fakeProvider{err: errRefused}, &fakeProvider{text: "never asked"}, "", 0},
		{"claude is unclear", &fakeProvider{text: " Unclear.\n"}, &fakeProvider{text: "never asked"}, "", 0},
		{"both fail", &fakeProvider{err: errNoAnswer}, &fakeProvider{err: errNoAnswer}, "", 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := newExplainer(tt.claude, tt.router)
			got := e.Explain(t.Context(), request("api", "boom"))
			assert.Equal(t, tt.want, got.Text)
			assert.True(t, got.ReusedFrom.IsZero())
			assert.EqualValues(t, 1, tt.claude.calls.Load())
			assert.Equal(t, tt.wantRouterCalls, tt.router.calls.Load())
		})
	}
}

func TestNewNeedsAKey(t *testing.T) {
	assert.Nil(t, New(Config{ClaudeModel: "claude-opus-5-5"}))
	e := New(Config{OpenRouterAPIKey: "k", OpenRouterModel: "m"})
	require.NotNil(t, e)
	require.Len(t, e.providers, 1)
	assert.IsType(t, &openRouter{}, e.providers[0], "with only an OpenRouter key it is used directly")
}

func TestExplainLongAnswerIsCut(t *testing.T) {
	e := newExplainer(&fakeProvider{text: strings.Repeat("word ", 300)})
	assert.LessOrEqual(t, len(e.Explain(t.Context(), request("api", "boom")).Text), maxExplanation)
}

func TestReuse(t *testing.T) {
	now := time.Date(2026, 10, 13, 14, 2, 0, 0, time.UTC)
	p := &fakeProvider{text: "Cause: a. Check: b."}
	e := newExplainer(p)
	e.now = func() time.Time { return now }

	var contextReads int
	req := request("ghcr.io/ternakklip/api:1.4.2", `{"status":500,"req":"0aece9f2-d3f2-485f-9b48-697faa6bcd9a"}`)
	req.Context = func(context.Context) ([]Line, []Line) {
		contextReads++
		return nil, nil
	}
	first := e.Explain(t.Context(), req)
	require.Equal(t, "Cause: a. Check: b.", first.Text)

	now = now.Add(time.Hour)
	again := request("ghcr.io/ternakklip/api:1.5.0@sha256:abc", `{"status":500,"req":"6f1c2b3a-0000-4000-8000-000000000001"}`)
	again.Context = req.Context
	got := e.Explain(t.Context(), again)
	assert.Equal(t, "Cause: a. Check: b.", got.Text, "a new tag and new ids are the same error")
	assert.Equal(t, now.Add(-time.Hour), got.ReusedFrom)
	assert.EqualValues(t, 1, p.calls.Load())
	assert.Equal(t, 1, contextReads, "a reused answer reads no context")

	e.Explain(t.Context(), request("ghcr.io/ternakklip/worker:1.4.2", `{"status":500}`))
	assert.EqualValues(t, 2, p.calls.Load(), "another image is another error")

	now = now.Add(reuseTTL)
	e.Explain(t.Context(), req)
	assert.EqualValues(t, 3, p.calls.Load(), "answers expire")
}

func TestReuseKeepsUnclearButNotFailures(t *testing.T) {
	unclearProvider := &fakeProvider{text: "UNCLEAR"}
	e := newExplainer(unclearProvider)
	e.Explain(t.Context(), request("api", "boom"))
	got := e.Explain(t.Context(), request("api", "boom"))
	assert.Empty(t, got.Text)
	assert.EqualValues(t, 1, unclearProvider.calls.Load())

	failing := &fakeProvider{err: errNoAnswer}
	e = newExplainer(failing)
	e.Explain(t.Context(), request("api", "boom"))
	e.Explain(t.Context(), request("api", "boom"))
	assert.EqualValues(t, 2, failing.calls.Load(), "the next group tries again")
}

func TestReuseEvictsTheOldest(t *testing.T) {
	now := time.Date(2026, 10, 13, 0, 0, 0, 0, time.UTC)
	e := newExplainer(&fakeProvider{text: "Cause: a. Check: b."})
	e.now = func() time.Time { return now }
	for i := range reuseMax + 1 {
		now = now.Add(time.Second)
		e.remember(fmt.Sprintf("key %d", i), "x")
	}
	assert.Len(t, e.answers, reuseMax)
	assert.NotContains(t, e.answers, "key 0")
	assert.Contains(t, e.answers, fmt.Sprintf("key %d", reuseMax))
}

func TestHourlyCap(t *testing.T) {
	now := time.Date(2026, 10, 13, 0, 0, 0, 0, time.UTC)
	p := &fakeProvider{text: "Cause: a. Check: b."}
	e := newExplainer(p)
	e.now = func() time.Time { return now }

	for i := range hourlyCap + 1 {
		e.Explain(t.Context(), request("api", fmt.Sprintf("error kind %c%c", 'a'+i/26, 'a'+i%26)))
	}
	assert.EqualValues(t, hourlyCap, p.calls.Load())

	got := e.Explain(t.Context(), request("api", "error kind aa"))
	assert.NotEmpty(t, got.Text, "a reused answer doesn't need the cap")

	now = now.Add(time.Hour)
	e.Explain(t.Context(), request("api", "a brand new error"))
	assert.EqualValues(t, hourlyCap+1, p.calls.Load(), "the cap resets every hour")
}

func TestExplainSendsTheBuiltInput(t *testing.T) {
	p := &fakeProvider{text: "Cause: a. Check: b."}
	req := request("api", "dial tcp 10.0.0.7:5432: connection refused")
	req.Context = func(context.Context) ([]Line, []Line) {
		return []Line{{Text: "starting job", Offset: -time.Second}}, nil
	}
	newExplainer(p).Explain(t.Context(), req)
	assert.Contains(t, p.input, "dial tcp <ip>:5432: connection refused")
	assert.Contains(t, p.input, "-1.0s starting job")
}
