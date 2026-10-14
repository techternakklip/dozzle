package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

const openRouterURL = "https://openrouter.ai/api/v1/chat/completions"

type openRouter struct {
	apiKey string
	model  string
	url    string
	client *http.Client
}

func newOpenRouter(apiKey, model, url string) *openRouter {
	return &openRouter{apiKey: apiKey, model: model, url: url, client: &http.Client{}}
}

type openRouterResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Content json.RawMessage `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64    `json:"prompt_tokens"`
		CompletionTokens int64    `json:"completion_tokens"`
		Cost             *float64 `json:"cost"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func (o *openRouter) explain(ctx context.Context, system, input string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, providerTimeout)
	defer cancel()

	body, err := json.Marshal(map[string]any{
		"model": o.model,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": input},
		},
		"max_completion_tokens": maxTokens,
		// Only route to providers that neither store nor train on prompts.
		"provider": map[string]string{"data_collection": "deny"},
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+o.apiKey)
	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := o.client.Do(req)
	if err != nil {
		log.Warn().Err(err).Str("model", o.model).Msg("OpenRouter call failed")
		return "", err
	}
	defer resp.Body.Close()

	var out openRouterResponse
	decodeErr := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&out)
	// OpenRouter reports a provider's failure inside an HTTP 200, so the body
	// decides as much as the status does.
	if resp.StatusCode/100 != 2 || decodeErr != nil || out.Error != nil || len(out.Choices) == 0 {
		event := log.Warn()
		if needsPerson(resp.StatusCode) {
			event = log.Error()
		}
		if out.Error != nil {
			event = event.Str("error", out.Error.Message)
		}
		event.Int("status", resp.StatusCode).Str("model", o.model).Msg("OpenRouter call failed")
		return "", errNoAnswer
	}

	choice := out.Choices[0]
	text := contentText(choice.Message.Content)
	log.Info().
		Str("provider", "openrouter").
		Str("model", out.Model).
		Int64("inputTokens", out.Usage.PromptTokens).
		Int64("outputTokens", out.Usage.CompletionTokens).
		Func(func(e *zerolog.Event) {
			if out.Usage.Cost != nil {
				e.Float64("cost", *out.Usage.Cost)
			}
		}).
		Str("finishReason", choice.FinishReason).
		Dur("took", time.Since(start)).
		Msg("AI call")

	if choice.FinishReason != "stop" || strings.TrimSpace(text) == "" {
		return "", errNoAnswer
	}
	return text, nil
}

// contentText reads a message's content, which is either a string or a list of parts.
func contentText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}
