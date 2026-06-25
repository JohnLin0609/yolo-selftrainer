Feature: cross-provider benchmark produces a comparable table from events.jsonl
  Goal: replace the README "Reliability" star ratings with reproducible data.
  Each scenario fabricates events.jsonl for one or more providers — no real
  LLM calls, no real training. The pipeline under test is the pure pair
  `aggregate_from_events` + `render_comparison_table`. End-to-end orchestration
  (scaffold → start_agent.sh → wait → aggregate) is a manual-smoke concern.

  Background:
    Given a temporary benchmark workspace

  Scenario: three providers with complete events produce a sorted comparison table
    Given a benchmarked provider "anthropic" model "claude-opus-4-7" with fixture "happy_anthropic"
    And a benchmarked provider "openai" model "gpt-4o" with fixture "happy_openai"
    And a benchmarked provider "gemini" model "gemini-2.5-pro" with fixture "no_test_no_cost_gemini"
    When the benchmark report is rendered
    Then the report contains exactly 3 data rows
    And the data rows appear in this provider order: anthropic, openai, gemini
    And the row for provider "anthropic" has val mAP cell "0.7800"
    And the row for provider "anthropic" has test mAP cell "0.7100"
    And the row for provider "gemini" has test mAP cell em-dash
    And the row for provider "gemini" has LLM cost cell "$0.00"
    And every row has halts cell "0"
    And the report includes the footnote line explaining the columns

  Scenario: a provider whose run hit the circuit breaker is included with its halt count
    Given a benchmarked provider "ollama" model "qwen2.5:32b" with fixture "halted_ollama"
    When the benchmark report is rendered
    Then the row for provider "ollama" has halts cell "1"
    And the report contains a halt-reasons section that names "circuit-breaker-agent" for "ollama"

  Scenario: aggregator numbers line up with the raw events file
    Given a benchmarked provider "anthropic" model "claude-opus-4-7" with fixture "two_rounds_durations_anthropic"
    When aggregate_from_events is called for that provider
    Then the resulting row's total_wall_sec equals 515
    And the resulting row's circuit_breaker_trips equals 0
