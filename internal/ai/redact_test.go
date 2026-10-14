package ai

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRedact(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			"gin path loses its query string",
			`[GIN] 2026/10/13 - 14:04:57 | 200 |     345.646µs |      172.18.0.1 | GET      "/api/v1/login?refresh_token=abc.def&next=/home"`,
			`[GIN] 2026/10/13 - 14:04:57 | 200 |     345.646µs |      172.18.0.1 | GET      "/api/v1/login?REDACTED"`,
		},
		{"url fragment", "callback https://x.example/cb#access_token=abc", "callback https://x.example/cb?REDACTED"},
		{"dsn credentials", "failed to connect database: postgres://admin:hunter2@db:5432/app", "failed to connect database: postgres://db:5432/app"},
		{"an @ in the password", "dial postgres://app:p@ss@db:5432/app", "dial postgres://db:5432/app"},
		{"an @ in the query is not a password", "open https://x.example/cb?ref=a@b.example", "open https://x.example/cb?REDACTED"},
		{"secret pair", "refresh_token=abc123 user=7", "refresh_token=REDACTED user=7"},
		{"secret pair with colon", "client_secret: s3cr3t, next", "client_secret: REDACTED, next"},
		{"secret pair in json", `{"password":"p\"w","name":"x"}`, `{"password":"REDACTED","name":"x"}`},
		{"bearer header", "Authorization: Bearer abc.def-ghi", "Authorization: REDACTED"},
		{"bearer alone", "sent Bearer abc.def-ghi upstream", "sent Bearer REDACTED upstream"},
		{"bearer in prose", "missing bearer token", "missing bearer token"},
		{"api key header", "header X-Api-Key: sk-live-abc123 sent", "header X-Api-Key: REDACTED sent"},
		{"api key in json", `{"x-api-key":"sk-live-abc123"}`, `{"x-api-key":"REDACTED"}`},
		{"closing paren survives", "access denied (using password: YES)", "access denied (using password: REDACTED)"},
		{"jwt", "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl rejected", "token REDACTED_JWT rejected"},
		{"long digits keep the last four", "account 1234567890123456 declined", "account ******3456 declined"},
		{"short numbers stay", "status 500 after 5003ms", "status 500 after 5003ms"},
		{"email keeps first letter and domain", "sent to budi.santoso@example.co.id", "sent to b***@example.co.id"},
		{
			"gorm sql values",
			`[3.784ms] [rows:0] SELECT * FROM "users" WHERE email = 'a@b.com' AND name = 'O''Brien' LIMIT 1`,
			`[3.784ms] [rows:0] SELECT * FROM "users" WHERE email = '?' AND name = '?' LIMIT 1`,
		},
		{"apostrophes outside sql stay", "DELETE /api/v1/clips: can't find clip 'x'", "DELETE /api/v1/clips: can't find clip 'x'"},
		{"apostrophe before the sql", "can't run SELECT * FROM users WHERE name = 'hunter2'", "can't run SELECT * FROM users WHERE name = '?'"},
		{
			"sql literals line by line",
			"query failed\nSELECT * FROM t WHERE a = 'x\nb'\nit's 'fine'",
			"query failed\nSELECT * FROM t WHERE a = '?'\nb'\nit's 'fine'",
		},
		{"frame keeps its line number", `"at":"usecase/auth/token_store.go:62 (*Store).Get"`, `"at":"usecase/auth/token_store.go:62 (*Store).Get"`},
		{
			"nothing sensitive is unchanged",
			`{"level":"error","msg":"500 CAMPAIGN_FETCH_FAILED GET /api/v1/campaigns/:public_id","status":500,"req":"0aece9f2-d3f2-485f-9b48-697faa6bcd9a","ip":"172.20.0.2"}`,
			`{"level":"error","msg":"500 CAMPAIGN_FETCH_FAILED GET /api/v1/campaigns/:public_id","status":500,"req":"0aece9f2-d3f2-485f-9b48-697faa6bcd9a","ip":"172.20.0.2"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, Redact(tt.in))
		})
	}
}

// The model's copy also masks ids and addresses; an alert's copy keeps the
// UUID so the line can be searched for in Dozzle.
func TestModelCopyMasksIDs(t *testing.T) {
	line := `{"req":"0aece9f2-d3f2-485f-9b48-697faa6bcd9a","ip":"172.20.0.2","peer":"fe80::1","at":"14:05:12"}`
	assert.Equal(t, `{"req":"<uuid>","ip":"<ip>","peer":"<ip>","at":"14:05:12"}`, maskIDs(Redact(line)))
	assert.Contains(t, Redact(line), "0aece9f2-d3f2-485f-9b48-697faa6bcd9a")

	numeric := "req 6f1c2b3a-0000-4000-8000-000000000001 failed"
	assert.Equal(t, numeric, Redact(numeric), "the digit rule leaves a UUID whole")

	assert.Equal(t, "dial tcp [<ip>]:5432: timeout", maskIDs("dial tcp [2001:db8::1]:5432: timeout"))
	for _, scope := range []string{"std::bad_alloc", "PG::ConnectionBad", "SELECT $1::uuid", "created_at::date"} {
		assert.Equal(t, scope, maskIDs(scope), "a scope inside a word is not an address")
	}

	// Masking an id must not hide the query string behind it from the redactor.
	gin := `GET "/api/v1/campaigns/0aece9f2-d3f2-485f-9b48-697faa6bcd9a?invite=Xk29fQ&otp=482913"`
	assert.Equal(t, "+0.0s GET \"/api/v1/campaigns/<uuid>?REDACTED\"\n", formatLine(Match{Text: gin, Count: 1}, maxContextLine))
}
