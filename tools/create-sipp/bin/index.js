#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { createInterface } from 'node:readline/promises';
import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';

const templateDir = fileURLToPath(new URL('../template', import.meta.url));
const textExtensions = new Set(['.css', '.html', '.js', '.json', '.md', '.ts']);
const validPackageName = /^[a-z0-9][a-z0-9._-]*$/;

function printHelp() {
  console.log(`
Usage:
  npm create @sipphq/sipp@latest [project-name] [options]
  npx @sipphq/create-sipp@latest [project-name] [options]

Options:
  -y, --yes     Skip prompts and use defaults
  -h, --help    Display this help message
`);
}

function copyDir(src, dest, replacements = {}) {
  fs.mkdirSync(dest, { recursive: true });
  const entries = fs.readdirSync(src, { withFileTypes: true });

  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destinationName =
      entry.name === 'gitignore' ? '.gitignore' : entry.name;
    const destPath = path.join(dest, destinationName);

    if (entry.isDirectory()) {
      copyDir(srcPath, destPath, replacements);
    } else {
      const content = fs.readFileSync(srcPath);
      if (textExtensions.has(path.extname(entry.name))) {
        let text = content.toString('utf8');
        for (const [key, value] of Object.entries(replacements)) {
          text = text.replaceAll(key, value);
        }
        fs.writeFileSync(destPath, text, 'utf8');
      } else {
        fs.writeFileSync(destPath, content);
      }
    }
  }
}

async function main() {
  const { values, positionals } = parseArgs({
    args: process.argv.slice(2),
    options: {
      help: { type: 'boolean', short: 'h', default: false },
      yes: { type: 'boolean', short: 'y', default: false },
    },
    allowPositionals: true,
    strict: true,
  });

  if (values.help) {
    printHelp();
    return 0;
  }

  const isYes = values.yes;
  if (positionals.length > 1) {
    throw new Error('Expected at most one project name.');
  }

  const targetArg = positionals[0];
  let rl = null;

  try {
    console.log('\n⚡ Welcome to create-sipp!\n');

    let projectName = targetArg;
    if (!projectName) {
      if (isYes) {
        projectName = 'sipp-app';
      } else {
        rl = createInterface({ input: process.stdin, output: process.stdout });
        const answer = await rl.question('Project name (default: sipp-app): ');
        projectName = answer.trim() || 'sipp-app';
      }
    }

    const targetDir = path.resolve(process.cwd(), projectName);
    const packageName = path.basename(targetDir);
    if (!validPackageName.test(packageName)) {
      throw new Error(
        `Project directory name "${packageName}" must be a lowercase npm package name.`,
      );
    }

    if (fs.existsSync(targetDir)) {
      const existingFiles = fs.readdirSync(targetDir);
      if (existingFiles.length > 0) {
        throw new Error(`Directory "${projectName}" is not empty.`);
      }
    }

    console.log(`\nScaffolding Sipp project in ${targetDir}...`);

    const replacements = {
      '{{PROJECT_NAME}}': packageName,
    };

    copyDir(templateDir, targetDir, replacements);

    console.log(`
✅ Project created successfully!

Next steps:
  cd ${projectName}
  npm install
  npm run dev

Cross-Origin Isolation (COOP & COEP) headers are pre-configured in vite.config.ts.
Happy building with browser-native WebGPU AI! ⚡
`);
    return 0;
  } finally {
    rl?.close();
  }
}

main()
  .then((exitCode) => {
    process.exitCode = exitCode;
  })
  .catch((error) => {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`\nError scaffolding project: ${message}`);
    process.exitCode = 1;
  });
