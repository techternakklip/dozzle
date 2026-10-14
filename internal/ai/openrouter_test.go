package ai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func openRouterChoice(finish, content string) string {
	return `{"model":"anthropic/claude-opus-5.5","choices":[{"finish_reason":"` + finish + `","message":{"content":` + content + `}}],
		"usage":{"prompt_tokens":2100,"completion_tokens":95,"cost":0.0103}}`
}

func TestOpenRouterRequestAndAnswer(t *testing.T) {
	logs := captureLogs(t)
	var auth string
	var sent map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		raw, _ := io.ReadAll(r.Body)
		json.Unmarshal(raw, &sent)
		io.WriteString(w, openRouterChoice("stop", `"Cause: x. Check: y."`))
	}))
	defer srv.Close()

	text, err := newOpenRouter(testKey, "anthropic/claude-opus-5.5", srv.URL).explain(t.Context(), "system", "lines")
	require.NoError(t, err)
	assert.Equal(t, "Cause: x. Check: y.", text)
	assert.Equal(t, "Bearer "+testKey, auth)
	assert.Equal(t, "anthropic/claude-opus-5.5", sent["model"])
	assert.EqualValues(t, maxTokens, sent["max_completion_tokens"])
	assert.Equal(t, map[string]any{"data_collection": "deny"}, sent["provider"])
	assert.NotContains(t, sent, "reasoning")
	assert.Contains(t, logs.String(), `"cost":0.0103`)
	assert.Contains(t, logs.String(), `"inputTokens":2100`)
	assert.NotContains(t, logs.String(), testKey)
}

func TestOpenRouterContentParts(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, openRouterChoice("stop", `[{"type":"text","text":"Cause: x. "},{"type":"text","text":"Check: y."}]`))
	}))
	defer srv.Close()

	text, err := newOpenRouter(testKey, "m", srv.URL).explain(t.Context(), "system", "lines")
	require.NoError(t, err)
	assert.Equal(t, "Cause: x. Check: y.", text)
}

func TestOpenRouterFailures(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
	}{
		{"error inside a 200", http.StatusOK, `{"error":{"code":502,"message":"provider returned error"}}`},
		{"out of credits", http.StatusPaymentRequired, `{"error":{"code":402,"message":"insufficient credits"}}`},
		{"unavailable", http.StatusServiceUnavailable, `{"error":{"code":503,"message":"no provider available"}}`},
		{"cut off", http.StatusOK, openRouterChoice("length", `"Cause: the"`)},
		{"empty", http.StatusOK, openRouterChoice("stop", `""`)},
		{"not json", http.StatusOK, `<html>gateway</html>`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logs := captureLogs(t)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
				io.WriteString(w, tt.body)
			}))
			defer srv.Close()

			_, err := newOpenRouter(testKey, "m", srv.URL).explain(t.Context(), "system", "lines")
			assert.ErrorIs(t, err, errNoAnswer)
			assert.NotContains(t, logs.String(), testKey)
		})
	}
}
