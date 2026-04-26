# ANES Data Process

Files:
- `parse_anes_questionnaire.py`: parses the ANES questionnaire PDF into JSON.

Default design:
- Pre-election survey items are used for persona prompts.
- Post-election vote (`V242067`) is treated as an outcome target, not a prompt feature.
- Pre-election two-way preference (`V241049`) is kept as an optional baseline/seeding variable and excluded from the default predictive prompt.
- Race release variables `V241500a-e` are excluded because in the uploaded CSV they are all `-3`.
- Age is excluded because there is no clean pre-election self-reported age variable in the questionnaire+CSV combination here.
