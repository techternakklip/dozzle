package ai

import (
	"regexp"
	"strings"
)

// The rules mirror the API's own RedactLogText (log.md §3). They run on every
// line that leaves the box, because the API leaves gin query strings, GORM SQL
// values and plain lines unredacted.
var (
	// Up to the last @ before the host, since a password may hold one.
	userinfoRe = regexp.MustCompile(`\b([a-zA-Z][a-zA-Z0-9+.-]*://)[^\s/?#"']+@`)
	queryRe    = regexp.MustCompile(`((?:\b[a-zA-Z][a-zA-Z0-9+.-]*://|/)[^\s"'?#<>]*)[?#][^\s"'<>]*`)
	jwtRe      = regexp.MustCompile(`\beyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]*`)
	bearerRe   = regexp.MustCompile(`(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}`)
	// A secret-named key in name=value, name: value or JSON form, with its value
	// quoted or bare. "Bearer x" counts as one bare value. Keys hold no dots, so
	// a frame such as token_store.go:42 keeps its line number.
	secretKey      = `(?i)("?[a-z0-9_-]*(?:token|secret|password|passwd|api[_-]?key|signature|authorization|cookie)[a-z0-9_-]*"?\s*[:=]\s*)`
	secretQuotedRe = regexp.MustCompile(secretKey + `"(?:[^"\\]|\\.)*"`)
	secretBareRe   = regexp.MustCompile(secretKey + `(?:(?:bearer|basic)\s+)?[^\s,;&"'})\]]+`)
	sqlRe          = regexp.MustCompile(`(?i)\b(?:select\b.*\bfrom|insert\s+into|update\b.*\bset|delete\s+from)\b`)
	sqlLiteralRe   = regexp.MustCompile(`'(?:[^']|'')*(?:'|$)`) // one still open masks to the line end
	emailRe        = regexp.MustCompile(`\b([a-zA-Z0-9])[a-zA-Z0-9._%+-]*@([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,})\b`)
	// UUIDs match first, so the digit rule never cuts into one that ends in digits.
	longDigitsRe = regexp.MustCompile(uuidPattern + `|\d{10,}`)
)

// redactSQL masks the literals GORM prints inline. Only from the SQL keyword
// on, line by line: before it, a quote is usually an apostrophe.
func redactSQL(s string) string {
	if !sqlRe.MatchString(s) {
		return s
	}
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		if loc := sqlRe.FindStringIndex(line); loc != nil {
			lines[i] = line[:loc[0]] + sqlLiteralRe.ReplaceAllString(line[loc[0]:], "'?'")
		}
	}
	return strings.Join(lines, "\n")
}

// Redact masks secrets and personal data in a log line: URL queries and
// credentials, secret-named values, bearer tokens and JWTs, SQL literals,
// emails, and long digit runs such as bank or phone numbers. Ids such as
// UUIDs stay, so a line quoted in an alert can still be searched for.
func Redact(s string) string {
	s = userinfoRe.ReplaceAllString(s, "$1")
	s = queryRe.ReplaceAllString(s, "$1?REDACTED")
	s = jwtRe.ReplaceAllString(s, "REDACTED_JWT")
	s = bearerRe.ReplaceAllString(s, "Bearer REDACTED")
	s = secretQuotedRe.ReplaceAllString(s, `${1}"REDACTED"`)
	s = secretBareRe.ReplaceAllString(s, "${1}REDACTED")
	s = redactSQL(s)
	s = emailRe.ReplaceAllString(s, "$1***@$2")
	return longDigitsRe.ReplaceAllStringFunc(s, func(m string) string {
		if strings.Contains(m, "-") {
			return m
		}
		return "******" + m[len(m)-4:]
	})
}
