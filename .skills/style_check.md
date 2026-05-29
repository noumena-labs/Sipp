# Style Checker Skill

You are an expert code reviewer. When instructed to check the style of the codebase, or prior to finishing any coding task, follow these steps to ensure the code adheres to the project's strict style guidelines.

## Execution Steps

1. **Get the Changes**: Run a command like `git diff` or `git diff --cached` to identify the files and lines that have been modified.
2. **Review the Guidance**: Read the monorepo's canonical style rules located at `.skills/style_guidance.md`.
3. **Analyze the Diff**: Compare the modified code against the style rules. Pay particular attention to:
    - **Rust**: Ensure custom error types (`thiserror`) are used in libraries, no naked `unwrap()` or `expect()` in production paths, and `//!` or `///` documentation is present.
    - **TypeScript**: Ensure NodeNext `.js` import extensions are present, arrays/objects are `readonly` where applicable, and no implicit `any` exists.
4. **Fix Issues**: If you detect violations, immediately use your file editing tools (e.g. `replace_file_content` or `multi_replace_file_content`) to correct the code.
5. **Report**: Summarize any style corrections you made to the user. If the code was perfectly compliant, explicitly state that the style check passed.
