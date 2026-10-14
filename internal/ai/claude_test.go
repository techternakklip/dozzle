package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testKey = "sk-ant-test-0123456789"

// captureLogs sends every log line to a buffer for the rest of the test.
func captureLogs(t *testing.T) *bytes.Buffer {
	var buf bytes.Buffer
	logger, level := log.Logger, zerolog.GlobalLevel()
	log.Logger = zerolog.New(&buf)
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
	t.Cleanup(func() {
		log.Logger = logger
		zerolog.SetGlobalLevel(level)
	})
	return &buf
}

func claudeServer(t *testing.T, status int, body string) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("request-id", "req_test")
		w.WriteHeader(status)
		io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func claudeMessage(stopReason, text, usage string) string {
	return `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5",
		"content":[{"type":"thinking","thinking":"hmm","signature":"sig"},{"type":"text","text":` + quote(text) + `}],
		"stop_reason":"` + stopReason + `","stop_sequence":null,"usage":` + usage + `}`
}

func quote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

const plainUsage = `{"input_tokens":2100,"output_tokens":90,"output_tokens_details":{"thinking_tokens":60}}`

func TestClaudeRequestAndAnswer(t *testing.T) {
	// A stray environment must not redirect calls or add credentials.
	t.Setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "someone-elses-token")
	logs := captureLogs(t)

	var got *http.Request
	var sent map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Clone(context.Background())
		raw, _ := io.ReadAll(r.Body)
		require.NoError(t, json.Unmarshal(raw, &sent))
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, claudeMessage("end_turn", "Cause: db down. Check: postgres.", plainUsage))
	}))
	defer srv.Close()

	text, err := newClaude(testKey, "claude-opus-5-5", srv.URL).explain(t.Context(), "system", "lines")
	require.NoError(t, err)
	assert.Equal(t, "Cause: db down. Check: postgres.", text, "thinking blocks are not part of the answer")

	assert.Equal(t, testKey, got.Header.Get("X-Api-Key"))
	assert.Empty(t, got.Header.Get("Authorization"))
	assert.Contains(t, got.Header.Get("Anthropic-Beta"), "server-side-fallback-2026-07-01")
	assert.Equal(t, "claude-opus-5-5", sent["model"])
	assert.Equal(t, "default", sent["fallbacks"])
	assert.Equal(t, map[string]any{"effort": "low"}, sent["output_config"])
	assert.NotContains(t, sent, "temperature")
	assert.NotContains(t, sent, "thinking")

	assert.Contains(t, logs.String(), `"inputTokens":2100`)
	assert.Contains(t, logs.String(), `"thinkingTokens":60`)
	assert.NotContains(t, logs.String(), testKey)
}

func TestClaudeFallbackOnlyWhereDocumented(t *testing.T) {
	var sent map[string]any
	var beta string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		beta = r.Header.Get("Anthropic-Beta")
		raw, _ := io.ReadAll(r.Body)
		json.Unmarshal(raw, &sent)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, claudeMessage("end_turn", "Cause: x. Check: y.", plainUsage))
	}))
	defer srv.Close()

	_, err := newClaude(testKey, "claude-haiku-5-5", srv.URL).explain(t.Context(), "system", "lines")
	require.NoError(t, err)
	assert.NotContains(t, sent, "fallbacks")
	assert.Empty(t, beta)
}

func TestClaudeSumsTokensAcrossTheRefusalFallback(t *testing.T) {
	logs := captureLogs(t)
	usage := `{"input_tokens":2000,"output_tokens":100,"output_tokens_details":{"thinking_tokens":70},
		"iterations":[{"type":"message","model":"claude-opus-5-5","input_tokens":2000,"output_tokens":20},
		{"type":"fallback_message","model":"claude-opus-5","input_tokens":2000,"output_tokens":100}]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, claudeMessage("end_turn", "Cause: x. Check: y.", usage))
	}))
	defer srv.Close()

	_, err := newClaude(testKey, "claude-opus-5-5", srv.URL).explain(t.Context(), "system", "lines")
	require.NoError(t, err)
	assert.Contains(t, logs.String(), `"inputTokens":4000`)
	assert.Contains(t, logs.String(), `"outputTokens":120`)
}

func TestClaudeFailures(t *testing.T) {
	apiError := func(kind string) string {
		return `{"type":"error","error":{"type":"` + kind + `","message":"nope"},"request_id":"req_test"}`
	}
	tests := []struct {
		name    string
		status  int
		body    string
		want    error // nil means any error that is not a refusal
		logged  string
		level   string
		refused bool
	}{
		{name: "credits exhausted", status: http.StatusPaymentRequired, body: apiError("billing_error"), logged: "Claude credits exhausted", level: "warn"},
		{name: "bad key", status: http.StatusUnauthorized, body: apiError("authentication_error"), logged: "Claude call failed", level: "error"},
		{name: "unknown model", status: http.StatusNotFound, body: apiError("not_found_error"), logged: "Claude call failed", level: "error"},
		{name: "overloaded", status: 529, body: apiError("overloaded_error"), logged: "Claude call failed", level: "warn"},
		{name: "refusal", status: http.StatusOK, body: claudeMessage("refusal", "", plainUsage), want: errRefused},
		{name: "max tokens", status: http.StatusOK, body: claudeMessage("max_tokens", "Cause: cut", plainUsage), want: errNoAnswer},
		{name: "empty answer", status: http.StatusOK, body: claudeMessage("end_turn", "  ", plainUsage), want: errNoAnswer},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logs := captureLogs(t)
			srv := claudeServer(t, tt.status, tt.body)

			_, err := newClaude(testKey, "claude-opus-5-5", srv.URL).explain(t.Context(), "system", "lines")
			require.Error(t, err)
			if tt.want != nil {
				assert.ErrorIs(t, err, tt.want)
			} else {
				assert.NotErrorIs(t, err, errRefused)
				assert.Contains(t, logs.String(), tt.logged)
				assert.Contains(t, logs.String(), `"level":"`+tt.level+`"`)
				assert.Contains(t, logs.String(), `"requestId":"req_test"`)
			}
			assert.NotContains(t, logs.String(), testKey)
		})
	}
}

func TestClaudeTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body) // the server only notices a client hanging up once the body is read
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	_, err := newClaude(testKey, "claude-opus-5-5", srv.URL).explain(ctx, "system", "lines")
	require.Error(t, err)
	assert.NotErrorIs(t, err, errRefused)
}
