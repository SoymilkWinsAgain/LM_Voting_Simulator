# ANES Data Process

Files:
- `parse_anes_questionnaire.py`: parses the ANES questionnaire PDF into JSON.

Default design:
- Pre-election survey items are used for persona prompts.
- Post-election vote (`V242067`) is treated as an outcome target, not a prompt feature.
- Pre-election two-way preference (`V241049`) is kept as an optional baseline/seeding variable and excluded from the default predictive prompt.
- Race release variables `V241500a-e` are excluded because in the uploaded CSV they are all `-3`.
- Age is available through `V241458x` and can be transformed into the same age groups used by CES.
- `V241023` is registration state, not full residence state. It may be used only as a weak matching feature and diagnostic, not as a hard residence-state block.
