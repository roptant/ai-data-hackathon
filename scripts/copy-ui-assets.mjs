// Copies static UI files next to the tsc output. Runs from ui/.
import { copyFileSync, mkdirSync } from "node:fs";

mkdirSync("dist", { recursive: true });
for (const file of ["index.html", "indicator.html", "styles.css", "indicator.css"]) {
  copyFileSync(file, `dist/${file}`);
}
