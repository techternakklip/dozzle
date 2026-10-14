package ai

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNormalize(t *testing.T) {
	same := [][2]string{
		{`{"req":"0aece9f2-d3f2-485f-9b48-697faa6bcd9a","ms":5003}`, `{"req":"6f1c2b3a-0000-4000-8000-000000000001","ms":12}`},
		{"dial tcp 10.0.0.7:5432: connection refused", "dial tcp 10.0.0.9:5432: connection refused"},
		{"dial tcp [2001:db8::1]:5432: timeout", "dial tcp [2001:db8::2]:5432: timeout"},
		{"commit deadbeef42 failed", "commit 0123abcdef failed"},
		{"[GIN] 2026/10/13 - 14:04:57 | 500 |  5.0s | GET \"/x\"", "[GIN] 2026/10/14 - 09:00:01 | 500 |  1.2s | GET \"/x\""},
		{"retry   in\t3s", "retry in 12s"},
	}
	for _, pair := range same {
		assert.Equal(t, Normalize(pair[0]), Normalize(pair[1]), "%q and %q", pair[0], pair[1])
	}

	assert.NotEqual(t, Normalize("duplicate key value violates unique constraint"), Normalize("canceling statement due to statement timeout"))

	// A long route pushes the cause far in, but two causes still differ, while
	// the request's own fields after it don't split one error in two.
	trace := func(cause, req, ip string, ms int) string {
		return fmt.Sprintf(`{"level":"error","msg":"500 CAMPAIGN_PARTICIPANT_FETCH_FAILED GET /api/v1/campaigns/:public_id/participants/:creator_id","status":500,"code":"CAMPAIGN_PARTICIPANT_FETCH_FAILED","cause":"failed to connect to host=postgres user=api database=app: %s","at":"repository/participant.go:91 (*Repo).find","req":"%s","ip":"%s","ms":%d}`, cause, req, ip, ms)
	}
	timeout := trace("dial error (timeout)", "0aece9f2-d3f2-485f-9b48-697faa6bcd9a", "172.20.0.2", 5003)
	assert.NotEqual(t, Normalize(timeout), Normalize(trace("server error (FATAL: sorry, too many clients already)", "0aece9f2-d3f2-485f-9b48-697faa6bcd9a", "172.20.0.2", 5003)))
	assert.Equal(t, Normalize(timeout), Normalize(trace("dial error (timeout)", "6f1c2b3a-0000-4000-8000-000000000001", "172.20.0.9", 4870)))
	assert.LessOrEqual(t, len(Normalize(strings.Repeat("word ", 100))), maxNormalized)
}

func TestTruncate(t *testing.T) {
	assert.Equal(t, "short", Truncate("short", 10))
	got := Truncate(strings.Repeat("é", 10), 10)
	assert.LessOrEqual(t, len(got), 10)
	assert.Equal(t, "ééé…", got, "never splits a character")
}

func match(text string, offset time.Duration, count int) Match {
	return Match{Text: text, Offset: offset, Count: count}
}

// The ternakklip samples from log.md §2.
const (
	ginHealth  = `[GIN] 2026/10/13 - 14:04:57 | 200 |     345.646µs |      172.18.0.1 | GET      "/health?platform=web"`
	ginFailure = `[GIN] 2026/10/13 - 14:05:12 | 500 |  5.003s |      172.18.0.1 | GET      "/api/v1/campaigns/3f0a"`
	gormError  = `/app/internal/repository/campaign_repository.go:280 failed to connect to host=postgres: dial error (timeout)`
)

func errorTrace(n int) string {
	return fmt.Sprintf(`{"level":"error","msg":"500 CAMPAIGN_FETCH_FAILED GET /api/v1/campaigns/:public_id","status":500,"code":"CAMPAIGN_FETCH_FAILED","cause":"failed to connect to host=postgres: dial error (timeout)","at":"repository/campaign_repository.go:280 (*CampaignRepository).findOne","req":"6f1c2b3a-0000-4000-8000-%012d","ms":5003,"trace":["repository/campaign_repository.go:280 (*CampaignRepository).findOne","usecase/campaign/get.go:61 (*GetCampaignUsecase).Execute","delivery/http/handler/campaign_handler.go:412 (*CampaignHandler).Get","delivery/http/router.go:88 Router.func1"]}`, n)
}

func TestBuildInput(t *testing.T) {
	var before []Line
	for i := range 30 {
		before = append(before, Line{Text: ginHealth, Offset: -time.Duration(30-i) * time.Second})
	}
	before = append(before,
		Line{Text: "\r", Offset: -1100 * time.Millisecond},
		Line{Text: gormError, Offset: -time.Second},
		Line{Text: ginFailure, Offset: -10 * time.Millisecond},
	)
	after := []Line{
		{Text: errorTrace(2), Offset: 0}, // the matched line itself, read back with the context
		{Text: "2026/10/13 14:05:13 [social_token_refresh] swept 4 credential(s)", Offset: time.Second},
	}
	req := Request{
		Container: "ternakklip_api",
		Image:     "ternakklip/api:1.4.2",
		Host:      "prod",
		Matched:   []Match{match(errorTrace(1), 0, 3)},
		Count:     3,
		Span:      42 * time.Second,
	}

	input := buildInput(req, before, after)

	assert.Contains(t, input, "Container: ternakklip_api\nImage: ternakklip/api:1.4.2\nHost: prod\nMatches: 3 in 42s")
	assert.Contains(t, input, "+0.0s ×3 "+maskIDs(Redact(errorTrace(1))), "the error-trace line reaches the model whole")
	assert.Contains(t, input, `"trace":[`)
	assert.Contains(t, input, "×30 "+maskIDs(Redact(ginHealth)), "30 health checks take one line")
	assert.Equal(t, 1, strings.Count(input, "/health"))
	assert.Contains(t, input, "-1.0s /app/internal/repository/campaign_repository.go:280 failed")
	assert.NotContains(t, input, "\r")
	assert.Equal(t, 1, strings.Count(input, "CAMPAIGN_FETCH_FAILED GET"), "a context line that repeats a match is dropped")
	assert.NotContains(t, input, "172.18.0.1")
	assert.NotContains(t, input, "6f1c2b3a")

	beforeAt := strings.Index(input, "Before the first match:")
	afterAt := strings.Index(input, "After the first match:")
	require.Positive(t, beforeAt)
	assert.Less(t, beforeAt, strings.Index(input, ginFailure[:20]))
	assert.Less(t, strings.Index(input, ginFailure[:20]), afterAt, "context stays in time order")
	assert.Contains(t, input[afterAt:], "+1.0s 2026/10/13 14:05:13 [social_token_refresh]")
}

func TestBuildInputLimits(t *testing.T) {
	t.Run("context lines are cut and stacks keep 15 lines", func(t *testing.T) {
		stack := make([]string, 40)
		for i := range stack {
			stack[i] = fmt.Sprintf("frame %d", i)
		}
		before := []Line{
			{Text: strings.Repeat("x", 900), Offset: -2 * time.Second},
			{Text: strings.Join(stack, "\n"), Offset: -time.Second, Stack: true},
		}
		input := buildInput(Request{Matched: []Match{match("boom", 0, 1)}, Count: 1}, before, nil)
		assert.Contains(t, input, strings.Repeat("x", maxContextLine-3)+"…\n")
		assert.NotContains(t, input, strings.Repeat("x", maxContextLine))
		assert.Contains(t, input, "frame 14\n")
		assert.NotContains(t, input, "frame 15")
	})

	t.Run("distinct matched lines past 6 KB are counted, most frequent first", func(t *testing.T) {
		var matched []Match
		for i := range 10 {
			matched = append(matched, match(fmt.Sprintf("error kind %c %s", 'a'+i, strings.Repeat("y", 1500)), 0, i+1))
		}
		input := buildInput(Request{Matched: matched, Count: 55}, nil, nil)
		assert.Contains(t, input, "×10 error kind j")
		assert.NotContains(t, input, "error kind a")
		assert.Contains(t, input, "+6 more distinct")
	})

	t.Run("the 8 KB cap drops the farthest context first", func(t *testing.T) {
		var before, after []Line
		for i := range 20 {
			before = append(before, Line{Text: fmt.Sprintf("before %c %s", 'a'+i, strings.Repeat("z", 400)), Offset: -time.Duration(20-i) * time.Second})
		}
		for i := range 10 {
			after = append(after, Line{Text: fmt.Sprintf("after %c %s", 'a'+i, strings.Repeat("z", 400)), Offset: time.Duration(i+1) * time.Second})
		}
		input := buildInput(Request{Matched: []Match{match(strings.Repeat("m", 2000), 0, 1)}, Count: 1}, before, after)
		assert.LessOrEqual(t, len(input), maxInput)
		assert.Contains(t, input, "before t ", "the nearest line before is kept")
		assert.Contains(t, input, "after a ", "the nearest line after is kept")
		assert.NotContains(t, input, "before a ", "the farthest line before is dropped")
	})
}
