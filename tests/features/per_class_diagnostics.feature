Feature: Per-class diagnostics surface in the prompt and the dataset stays read-only
  After each training round the framework parses per-class P/R/mAP and the
  confusion matrix, summarises them into the agent's next prompt, and (when
  a class has been worst for several consecutive rounds) prompts the agent
  to write a human-readable data-layer recommendation. The agent is NOT
  permitted to mutate the dataset — the Bash guard enforces this.

  Background:
    Given a fresh project directory

  Scenario: per-class diagnosis from the latest run reaches the next prompt
    Given a completed training run named "r1" at round 1
    And the per_class_metrics event for "r1" records class "scratch" with mAP50=0.12 and class "dent" with mAP50=0.88
    When build_prompt.py renders round 2 of 5
    Then the prompt contains a "Per-class diagnosis" section
    And the prompt names "scratch" as the weakest class
    And the prompt points the agent at the confusion_matrix.png file rather than embedding the numeric matrix

  Scenario: a class weak for N consecutive rounds triggers a data-layer recommendation block
    Given three consecutive runs where class "scratch" is the worst class each round
    When build_prompt.py renders round 4 of 5
    Then the prompt contains a "Persistent weakness detected" block flagging "scratch"
    And the block recommends labeling, samples, or augmentation review
    And the prompt explicitly tells the agent not to modify the dataset directory
    And the Bash guard rejects writing under the project's datasets directory
    And the Bash guard still allows reading the project's datasets directory
