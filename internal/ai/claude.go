package ai

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/rs/zerolog/log"
)

const claudeURL = "https://api.anthropic.com"

// fallbackModels are documented to accept fallbacks: "default", which reruns a
// declined request on another Claude model in the same call.
var fallbackModels = []string{"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}

type claude struct {
	client anthropic.Client
	model  string
}

func newClaude(apiKey, model, baseURL string) *claude {
	return &claude{
		model: model,
		client: anthropic.NewClient(
			// Otherwise the SDK also reads ANTHROPIC_* variables, profiles and
			// config files, any of which could redirect calls billed to our credits.
			option.WithoutEnvironmentDefaults(),
			option.WithBaseURL(baseURL),
			option.WithAPIKey(apiKey),
			option.WithHTTPClient(&http.Client{}),
			// OpenRouter is the retry.
			option.WithMaxRetries(0),
		),
	}
}

func (c *claude) explain(ctx context.Context, system, input string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, providerTimeout)
	defer cancel()

	params := anthropic.BetaMessageNewParams{
		Model:     c.model,
		MaxTokens: maxTokens,
		System:    []anthropic.BetaTextBlockParam{{Text: system}},
		Messages:  []anthropic.BetaMessageParam{anthropic.NewBetaUserMessage(anthropic.NewBetaTextBlock(input))},
		// Thinking is always on for Opus 5.5, so the lowest effort keeps it short.
		OutputConfig: anthropic.BetaOutputConfigParam{Effort: anthropic.BetaOutputConfigEffortLow},
	}
	if slices.Contains(fallbackModels, c.model) {
		params.Fallbacks = anthropic.BetaFallbacksParamOfDefault()
		params.Betas = []anthropic.AnthropicBeta{anthropic.AnthropicBetaServerSideFallback2026_07_01}
	}

	start := time.Now()
	msg, err := c.client.Beta.Messages.New(ctx, params)
	if err != nil {
		c.logError(err)
		return "", err
	}

	var text strings.Builder
	for _, block := range msg.Content {
		if block.Type == "text" {
			text.WriteString(block.Text)
		}
	}

	// With the refusal fallback, the top-level counts may only cover the model
	// that answered, so add up every attempt.
	in, out := msg.Usage.InputTokens, msg.Usage.OutputTokens
	if len(msg.Usage.Iterations) > 0 {
		in, out = 0, 0
		for _, it := range msg.Usage.Iterations {
			in += it.InputTokens
			out += it.OutputTokens
		}
	}
	log.Info().
		Str("provider", "claude").
		Str("model", msg.Model).
		Int64("inputTokens", in).
		Int64("outputTokens", out).
		Int64("thinkingTokens", msg.Usage.OutputTokensDetails.ThinkingTokens).
		Str("stopReason", string(msg.StopReason)).
		Dur("took", time.Since(start)).
		Msg("AI call")

	switch {
	case msg.StopReason == anthropic.BetaStopReasonRefusal:
		return "", errRefused
	case msg.StopReason != anthropic.BetaStopReasonEndTurn || strings.TrimSpace(text.String()) == "":
		return "", errNoAnswer
	}
	return text.String(), nil
}

func (c *claude) logError(err error) {
	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) {
		log.Warn().Err(err).Str("model", c.model).Msg("Claude call failed")
		return
	}
	event, msg := log.Warn(), "Claude call failed"
	if apiErr.Type() == anthropic.ErrorTypeBillingError {
		msg = "Claude credits exhausted"
	} else if needsPerson(apiErr.StatusCode) {
		event = log.Error()
	}
	event.Int("status", apiErr.StatusCode).Str("type", string(apiErr.Type())).Str("requestId", apiErr.RequestID).Str("model", c.model).Msg(msg)
}

// needsPerson is true for the statuses a retry can't fix, most likely a typo in
// the key or the model name.
func needsPerson(status int) bool {
	switch status {
	case http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound:
		return true
	}
	return false
}
