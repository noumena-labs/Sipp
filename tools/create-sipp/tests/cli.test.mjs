import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const cliPath = fileURLToPath(new URL('../bin/index.js', import.meta.url));

test('create-sipp CLI displays help with --help', () => {
  const output = execFileSync('node', [cliPath, '--help'], { encoding: 'utf8' });
  assert.match(output, /Usage:/);
  assert.match(output, /npm create @sipphq\/sipp@latest/);
  assert.match(output, /npx @sipphq\/create-sipp@latest/);
});

test('create-sipp scaffolds a complete runnable project', () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'create-sipp-test-'));
  const targetDir = path.join(tempDir, 'my-chat-app');

  try {
    const output = execFileSync('node', [cliPath, targetDir, '--yes'], {
      encoding: 'utf8',
      cwd: tempDir,
    });

    assert.match(output, /Project created successfully/);

    // Verify package.json
    const pkgPath = path.join(targetDir, 'package.json');
    assert.ok(fs.existsSync(pkgPath), 'package.json should exist');
    const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
    assert.equal(pkg.name, 'my-chat-app');
    assert.ok(pkg.dependencies['@sipphq/sipp'], 'should depend on @sipphq/sipp');

    // Verify vite.config.ts uses the package-owned isolation configuration.
    const viteConfigPath = path.join(targetDir, 'vite.config.ts');
    assert.ok(fs.existsSync(viteConfigPath), 'vite.config.ts should exist');
    const viteConfig = fs.readFileSync(viteConfigPath, 'utf8');
    assert.match(viteConfig, /@sipphq\/sipp\/vite/);
    assert.match(viteConfig, /sippViteConfig\(\)/);

    // Verify tsconfig.json
    assert.ok(fs.existsSync(path.join(targetDir, 'tsconfig.json')));
    assert.ok(fs.existsSync(path.join(targetDir, '.gitignore')));

    // Verify index.html & src
    assert.ok(fs.existsSync(path.join(targetDir, 'index.html')));
    assert.ok(fs.existsSync(path.join(targetDir, 'src/main.ts')));
    assert.ok(fs.existsSync(path.join(targetDir, 'src/style.css')));

    // Verify README.md has project name
    const readme = fs.readFileSync(path.join(targetDir, 'README.md'), 'utf8');
    assert.match(readme, /# my-chat-app/);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('create-sipp refuses to overwrite a non-empty directory', () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'create-sipp-existing-'));
  const targetDir = path.join(tempDir, 'existing-app');
  fs.mkdirSync(targetDir);
  fs.writeFileSync(path.join(targetDir, 'keep.txt'), 'keep');

  try {
    const result = spawnSync('node', [cliPath, targetDir], {
      encoding: 'utf8',
      cwd: tempDir,
    });

    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /is not empty/);
    assert.equal(fs.readFileSync(path.join(targetDir, 'keep.txt'), 'utf8'), 'keep');
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('create-sipp rejects an invalid npm package name', () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'create-sipp-invalid-'));

  try {
    const result = spawnSync('node', [cliPath, 'Invalid Name', '--yes'], {
      encoding: 'utf8',
      cwd: tempDir,
    });

    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /must be a lowercase npm package name/);
    assert.equal(fs.existsSync(path.join(tempDir, 'Invalid Name')), false);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});
