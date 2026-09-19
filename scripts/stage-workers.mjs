// Builds and stages Tauri sidecars on Windows, macOS, and Linux.
import { chmodSync, copyFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const workers = ["dictation-asr-worker", "dictation-privacy-worker"];
const dryRun = process.argv.includes("--dry-run");

function run(program, args, options = {}) {
  const result = spawnSync(program, args, {
    cwd: root,
    encoding: "utf8",
    ...options,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${program} exited with status ${result.status}`);
  }
  return result.stdout ?? "";
}

function hostTriple() {
  const line = run("rustc", ["-vV"]).split(/\r?\n/u).find((value) => value.startsWith("host: "));
  if (!line) throw new Error("rustc did not report a host target triple");
  return line.slice("host: ".length).trim();
}

const triple = process.env.TARGET_TRIPLE || hostTriple();
if (!/^[A-Za-z0-9_.-]+$/u.test(triple)) {
  throw new Error("TARGET_TRIPLE contains unsupported characters");
}
const executableSuffix = triple.includes("windows") ? ".exe" : "";
const sourceDirectory = join(root, "target", triple, "release");
const destinationDirectory = join(root, "src-tauri", "binaries");
const staged = workers.map((worker) => ({
  source: join(sourceDirectory, `${worker}${executableSuffix}`),
  destination: join(destinationDirectory, `${worker}-${triple}${executableSuffix}`),
}));

if (dryRun) {
  process.stdout.write(`${JSON.stringify({ triple, staged }, null, 2)}\n`);
  process.exit(0);
}

run(
  "cargo",
  [
    "build",
    "--release",
    "--locked",
    "--target",
    triple,
    "-p",
    workers[0],
    "-p",
    workers[1],
  ],
  { stdio: "inherit" },
);
mkdirSync(destinationDirectory, { recursive: true });
for (const { source, destination } of staged) {
  copyFileSync(source, destination);
  if (!executableSuffix) chmodSync(destination, 0o755);
}
process.stdout.write(`staged workers for ${triple}\n`);
