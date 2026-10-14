package ai

import (
	"cmp"
	"fmt"
	"net"
	"regexp"
	"slices"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	maxInput          = 8 << 10 // the whole model input
	maxMatchedSection = 6 << 10 // distinct matched lines, most frequent first
	maxMatchedLine    = 2048    // the API's own bound on its error-trace line
	maxContextLine    = 500
	maxStackLines     = 15
	maxBefore         = 20 // context lines before the first match, after collapsing
	maxAfter          = 10
	// maxNormalized reaches into a JSON error line's cause, while the request's
	// own fields that such lines put last mostly stay out of the shape.
	maxNormalized = 300
)

const uuidPattern = `\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b`

var (
	uuidRe      = regexp.MustCompile(uuidPattern)
	ipv4Re      = regexp.MustCompile(`\b(?:\d{1,3}\.){3}\d{1,3}\b`)
	ipv6Re      = regexp.MustCompile(`(?i)[0-9a-f]*:[0-9a-f:]*:[0-9a-f]*`)
	hexRe       = regexp.MustCompile(`(?i)\b(?:0x)?[0-9a-f]{8,}\b`)
	digitsRe    = regexp.MustCompile(`\d+`)
	whitespaceR = regexp.MustCompile(`\s+`)
)

// Normalize reduces a line to its shape, so repeats of one error that differ
// only in ids, addresses, numbers or times compare equal.
func Normalize(s string) string {
	s = maskIDs(s)
	s = hexRe.ReplaceAllStringFunc(s, func(m string) string {
		// A plain number is left to the digit rule, so 1234567 and 12345678 match.
		if strings.IndexFunc(m, func(r rune) bool { return r > '9' }) < 0 {
			return m
		}
		return "<hex>"
	})
	s = digitsRe.ReplaceAllString(s, "0")
	s = strings.TrimSpace(whitespaceR.ReplaceAllString(s, " "))
	return Truncate(s, maxNormalized)
}

// maskIDs replaces UUIDs and IP addresses, which cost tokens and say nothing
// about a cause.
func maskIDs(s string) string {
	s = uuidRe.ReplaceAllString(s, "<uuid>")
	s = ipv4Re.ReplaceAllString(s, "<ip>")

	// The pattern also finds times such as 14:05:12, which don't parse, and
	// scopes such as std::bad_alloc or $1::uuid, which sit inside a word.
	var b strings.Builder
	last := 0
	for _, loc := range ipv6Re.FindAllStringIndex(s, -1) {
		if isWordByte(s, loc[0]-1) || isWordByte(s, loc[1]) || net.ParseIP(s[loc[0]:loc[1]]) == nil {
			continue
		}
		b.WriteString(s[last:loc[0]] + "<ip>")
		last = loc[1]
	}
	return b.String() + s[last:]
}

func isWordByte(s string, i int) bool {
	if i < 0 || i >= len(s) {
		return false
	}
	c := s[i]
	return c == '_' || '0' <= c && c <= '9' || 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z'
}

// Truncate cuts s to at most n bytes without splitting a character, and marks the cut.
func Truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	cut := n - len("…")
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "…"
}

// buildInput turns a group into the text a model reads. Lines are collapsed
// before any limit is counted, so a run of health checks takes one slot.
func buildInput(req Request, before, after []Line) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Container: %s\nImage: %s\nHost: %s\n", req.Container, req.Image, req.Host)
	fmt.Fprintf(&b, "Matches: %d in %s\n\n", req.Count, req.Span.Round(time.Second))

	matched := slices.Clone(req.Matched)
	slices.SortStableFunc(matched, func(x, y Match) int { return y.Count - x.Count })
	skip := make(map[string]bool, len(matched))
	for _, m := range matched {
		skip[Normalize(m.Text)] = true
	}

	b.WriteString("Matched lines, most frequent first (offset from the first match):\n")
	used := 0
	for i, m := range matched {
		line := formatLine(m, maxMatchedLine)
		if i > 0 && used+len(line) > maxMatchedSection {
			fmt.Fprintf(&b, "+%d more distinct\n", len(matched)-i)
			break
		}
		b.WriteString(line)
		used += len(line)
	}

	// Nearest the first match first, so the farthest lines are the ones dropped.
	slices.Reverse(before)
	context := append(collapse(before, skip, maxBefore), collapse(after, skip, maxAfter)...)
	slices.SortStableFunc(context, func(x, y Match) int { return cmp.Compare(x.Offset.Abs(), y.Offset.Abs()) })
	const beforeHeading, afterHeading = "\nBefore the first match:\n", "\nAfter the first match:\n"
	budget := maxInput - b.Len() - len(beforeHeading) - len(afterHeading)
	var kept []Match
	for _, m := range context {
		if n := len(formatLine(m, maxContextLine)); n <= budget {
			budget -= n
			kept = append(kept, m)
		}
	}
	slices.SortStableFunc(kept, func(x, y Match) int { return cmp.Compare(x.Offset, y.Offset) })

	wroteBefore, wroteAfter := false, false
	for _, m := range kept {
		switch {
		case m.Offset < 0 && !wroteBefore:
			b.WriteString(beforeHeading)
			wroteBefore = true
		case m.Offset >= 0 && !wroteAfter:
			b.WriteString(afterHeading)
			wroteAfter = true
		}
		b.WriteString(formatLine(m, maxContextLine))
	}
	return b.String()
}

// collapse folds lines that normalize the same into the first of them with a
// count, and drops blank lines and lines that repeat a matched one.
func collapse(lines []Line, skip map[string]bool, limit int) []Match {
	var out []Match
	index := make(map[string]int)
	for _, l := range lines {
		if strings.TrimSpace(l.Text) == "" {
			continue
		}
		key := Normalize(l.Text)
		if skip[key] {
			continue
		}
		if i, ok := index[key]; ok {
			out[i].Count++
		} else if len(out) < limit {
			index[key] = len(out)
			out = append(out, Match{Line: l, Count: 1})
		}
	}
	return out
}

func formatLine(m Match, limit int) string {
	text := maskIDs(Redact(m.Text))
	if m.Stack {
		lines := strings.Split(text, "\n")
		for i := range lines {
			lines[i] = Truncate(lines[i], limit)
		}
		text = strings.Join(lines[:min(len(lines), maxStackLines)], "\n")
	} else {
		text = Truncate(text, limit)
	}
	count := ""
	if m.Count > 1 {
		count = fmt.Sprintf("×%d ", m.Count)
	}
	return fmt.Sprintf("%+.1fs %s%s\n", m.Offset.Seconds(), count, text)
}
