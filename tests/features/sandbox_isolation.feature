Feature: Sandboxed Bash cannot escape the project directory
  Once the sandbox runtime lands (approach 1 in the plan), every Bash
  invocation by the agent MUST be unable to mutate state outside its
  project root, read sibling projects, or reach the network. These
  scenarios describe the contract the implementation must satisfy.

  Today they are SKIPPED because scripts/sandbox.run_in_sandbox does
  not yet exist (conftest.py keys the skip on importlib.util.find_spec).
  Adding the module flips them to run automatically.

  Background:
    Given a sandboxed project at "sandbox_e2e"
    And a sibling project "other" must not be touched

  Scenario: command cannot delete sibling project
    When the in-sandbox command runs:
      """
      rm -rf ../other
      """
    Then the sibling project "other" still exists

  Scenario: command cannot write outside writable mounts
    When the in-sandbox command runs:
      """
      echo poison > /tmp/sandbox_escape_marker
      """
    Then the host file "/tmp/sandbox_escape_marker" does not exist

  Scenario: command cannot read sibling project's events.jsonl
    Given the sibling project "other" has secret content in its events.jsonl
    When the in-sandbox command runs:
      """
      cat ../other/events.jsonl
      """
    Then the in-sandbox command exits non-zero

  Scenario: writes within the project's runs dir succeed
    When the in-sandbox command runs:
      """
      mkdir -p runs/test && touch runs/test/marker
      """
    Then the in-sandbox command exits zero
    And the file "runs/test/marker" exists within the project

  Scenario: writes within the project root succeed
    When the in-sandbox command runs:
      """
      echo {} > next_params.json
      """
    Then the in-sandbox command exits zero
    And the file "next_params.json" exists within the project

  Scenario: framework code is mounted read-only
    When the in-sandbox command runs:
      """
      echo poison >> ../../scripts/event.py
      """
    Then the in-sandbox command exits non-zero

  Scenario: network is denied
    When the in-sandbox command runs:
      """
      curl -s --max-time 3 https://example.com >/dev/null
      """
    Then the in-sandbox command exits non-zero
