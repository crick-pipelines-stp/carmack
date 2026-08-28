# carmack
Single-cell multi-omic tools

## Development

### Git hooks

Version-controlled hooks live in `.githooks/`. Enable them once per clone:

```sh
git config core.hooksPath .githooks
```

The `pre-commit` hook auto-formats staged Python with `isort` then `black`
(both provided by the project's `carmack` conda env). Skip it for a single
commit with `git commit --no-verify`.
