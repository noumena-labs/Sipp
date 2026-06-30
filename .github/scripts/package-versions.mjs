import { readFileSync, writeFileSync } from 'node:fs';

const cargoPath = 'Cargo.toml';
const jsonPackagePaths = ['lib/web/package.json', 'lib/node/package.json'];
const pythonBackends = ['cuda', 'metal', 'vulkan'];

const command = process.argv[2];

if (command === 'resolve') {
  const version = resolveAlignedVersion();
  console.error(`Resolved aligned version ${version}`);
  console.log(version);
} else if (command === 'apply') {
  const packageVersion = requiredEnv('PACKAGE_VERSION');
  const pythonPackageVersion = requiredEnv('PYTHON_PACKAGE_VERSION');
  applyPackageVersions(packageVersion, pythonPackageVersion);
} else {
  throw new Error(`Unknown package version command: ${command ?? '(missing)'}`);
}

function resolveAlignedVersion() {
  const cargo = readFileSync(cargoPath, 'utf8');
  const pyproject = readFileSync('lib/python/pyproject.toml', 'utf8');
  const pythonBackendExtras = [
    ...pyproject.matchAll(/sipppy-backend-(?:cuda|metal|vulkan)==([^";\s]+)/g),
  ].map((match) => match[1]);
  const pythonBackendPackages = Object.fromEntries(
    pythonBackends.map((backend) => {
      const backendPyproject = readFileSync(
        `lib/python/backends/${backend}/pyproject.toml`,
        'utf8',
      );
      return [
        `pythonBackendPackage${backend}`,
        backendPyproject.match(/^version = "([^"]+)"/m)?.[1],
      ];
    }),
  );
  const versions = {
    cargoBindingDto: cargo.match(
      /^sipp-binding-dto = \{ path = "lib\/binding-dto", version = "([^"]+)"/m,
    )?.[1],
    cargoGateway: cargo.match(
      /^sipp-gateway = \{ path = "lib\/gateway", version = "([^"]+)"/m,
    )?.[1],
    cargoSipp: cargo.match(
      /^sipp = \{ path = "crates\/sipp", version = "([^"]+)"/m,
    )?.[1],
    cargoSys: cargo.match(
      /^sipp-sys = \{ path = "crates\/sys", version = "([^"]+)"/m,
    )?.[1],
    cargoWorkspace: cargo.match(
      /^\[workspace\.package\][\s\S]*?^version = "([^"]+)"/m,
    )?.[1],
    node: readJsonVersion('lib/node/package.json'),
    python: pyproject.match(/^version = "([^"]+)"/m)?.[1],
    ...pythonBackendPackages,
    web: readJsonVersion('lib/web/package.json'),
  };
  for (const [index, extraVersion] of pythonBackendExtras.entries()) {
    versions[`pythonBackendExtra${index}`] = extraVersion;
  }

  const missing = Object.entries(versions)
    .filter(([, value]) => value == null)
    .map(([name]) => name);
  if (missing.length > 0) {
    throw new Error(`Missing version field(s): ${missing.join(', ')}`);
  }

  const unique = [...new Set(Object.values(versions))];
  if (unique.length !== 1) {
    throw new Error(
      `Package versions are not aligned: ${JSON.stringify(versions)}`,
    );
  }

  const version = unique[0];
  if (!/^\d+\.\d+\.\d+$/.test(version)) {
    throw new Error(`Version must be a stable x.y.z value, got ${version}`);
  }
  return version;
}

function applyPackageVersions(packageVersion, pythonPackageVersion) {
  for (const path of jsonPackagePaths) {
    const pkg = JSON.parse(readFileSync(path, 'utf8'));
    pkg.version = packageVersion;
    writeFileSync(path, `${JSON.stringify(pkg, null, 2)}\n`);
  }

  const pyprojectPath = 'lib/python/pyproject.toml';
  const pyproject = readFileSync(pyprojectPath, 'utf8')
    .replace(/^version = ".*"$/m, `version = "${pythonPackageVersion}"`)
    .replace(
      /(sipppy-backend-(?:cuda|metal|vulkan)==)[^";\s]+/g,
      `$1${pythonPackageVersion}`,
    );
  writeFileSync(pyprojectPath, pyproject);

  for (const backend of pythonBackends) {
    const backendPyprojectPath = `lib/python/backends/${backend}/pyproject.toml`;
    const backendPyproject = readFileSync(backendPyprojectPath, 'utf8').replace(
      /^version = ".*"$/m,
      `version = "${pythonPackageVersion}"`,
    );
    writeFileSync(backendPyprojectPath, backendPyproject);
  }

  const cargo = readFileSync(cargoPath, 'utf8')
    .replace(/^version = ".*"$/m, `version = "${packageVersion}"`)
    .replace(
      /(sipp = \{ path = "crates\/sipp", version = )"[^"]+"/,
      `$1"${packageVersion}"`,
    )
    .replace(
      /(sipp-sys = \{ path = "crates\/sys", version = )"[^"]+"/,
      `$1"${packageVersion}"`,
    )
    .replace(
      /(sipp-gateway = \{ path = "lib\/gateway", version = )"[^"]+"/,
      `$1"${packageVersion}"`,
    )
    .replace(
      /(sipp-binding-dto = \{ path = "lib\/binding-dto", version = )"[^"]+"/,
      `$1"${packageVersion}"`,
    );
  writeFileSync(cargoPath, cargo);
}

function readJsonVersion(path) {
  return JSON.parse(readFileSync(path, 'utf8')).version;
}

function requiredEnv(name) {
  const value = process.env[name];
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}
