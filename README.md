# Payment Risk ML Pipeline

A small machine-learning project for estimating transaction risk from tabular payment data.

The project grew out of the kinds of patterns I investigate in payment support work: failed transactions, unusual amounts, repeated attempts, channel differences, and response-code behavior. It does **not** use employer or customer data. Examples are synthetic, and the pipeline is designed so a public dataset can be plugged in later.

The first goal is intentionally simple: build a reproducible baseline, avoid leakage, measure the right metrics for an imbalanced target, and keep the training path easy to inspect.
