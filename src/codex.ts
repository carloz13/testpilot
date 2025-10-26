import axios from "axios";
import fs from "fs";
import { performance } from "perf_hooks";
import { ICompletionModel } from "./completionModel";
import { trimCompletion } from "./syntax";

const defaultPostOptions = {
  max_tokens: 100, // maximum number of tokens to return
  temperature: 0, // sampling temperature; higher values increase diversity
  n: 5, // number of completions to return
  top_p: 1, // no need to change this
  model: "deepseek-coder:6.7b"
};
export type PostOptions = Partial<typeof defaultPostOptions>;

function getEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    console.error(`Please set the ${name} environment variable.`);
    process.exit(1);
  }
  return value;
}

export class Codex implements ICompletionModel {
  private readonly apiEndpoint: string;
  private readonly authHeaders: string;

  constructor(
    private readonly isStarCoder: boolean,
    private readonly instanceOptions: PostOptions = {}
  ) {
    this.apiEndpoint = this.isStarCoder
      ? getEnv("STARCODER_API_ENDPOINT")
      : getEnv("TESTPILOT_LLM_API_ENDPOINT");
    this.authHeaders = this.isStarCoder
      ? "{}"
      : getEnv("TESTPILOT_LLM_AUTH_HEADERS");
    console.log(
      `Using ${this.isStarCoder ? "StarCoder" : "GPT"} API at ${
        this.apiEndpoint
      }`
    );
  }

  /**
   * Query Codex for completions with a given prompt.
   *
   * @param prompt The prompt to use for the completion.
   * @param requestPostOptions The options to use for the request.
   * @returns A promise that resolves to a set of completions.
   */
  public async query(
    prompt: string,
    requestPostOptions: PostOptions = {}
  ): Promise<Set<string>> {
    const headers = {
      "Content-Type": "application/json",
      ...JSON.parse(this.authHeaders),
    };
    const options = {
      ...defaultPostOptions,
      // options provided to constructor override default options
      ...this.instanceOptions,
      // options provided to this function override default and instance options
      ...requestPostOptions,
    };

    performance.mark("codex-query-start");

    const postOptions = this.isStarCoder
      ? {
          inputs: prompt,
          parameters: {
            max_new_tokens: options.max_tokens,
            temperature: options.temperature || 0.01, // StarCoder doesn't allow 0
            n: options.n,
          },
        }
      : {
          prompt,
          ...options,
          model: options.model
        };

    const res = await axios.post(this.apiEndpoint, postOptions, { headers });

    performance.measure(
      `codex-query:${JSON.stringify({
        ...options,
        promptLength: prompt.length,
      })}`,
      "codex-query-start"
    );
    if (res.status !== 200) {
      throw new Error(
        `Request failed with status ${res.status} and message ${res.statusText}`
      );
    }
    if (!res.data) {
      throw new Error("Response data is empty");
    }
    // The endpoint may return several shapes:
    // - a JSON object with `choices[]` (OpenAI)
    // - a JSON object with `response` (Ollama-like)
    // - newline-delimited JSON chunks (streamed NDJSON) where each line is a JSON object with `response` fragments
    let json: any = res.data;

    // If we received a string that looks like NDJSON (many JSON objects separated by newlines), or
    // the endpoint returned an array of fragments, try to parse and assemble them into a single text
    if (typeof json === "string") {
      const lines = json.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
      if (lines.length > 1) {
        const fragments: any[] = [];
        for (const line of lines) {
          try {
            fragments.push(JSON.parse(line));
          } catch (e) {
            // ignore non-JSON lines
          }
        }
        if (fragments.length > 0) {
          const assembled = fragments
            .map((f) => {
              if (typeof f.response === "string") return f.response;
              if (typeof f.generated_text === "string") return f.generated_text;
              if (f.outputs && Array.isArray(f.outputs))
                return f.outputs.map((o: any) => o.text || o.content || "").join("");
              if (f.choices && Array.isArray(f.choices))
                return f.choices.map((c: any) => c.text || c.response || "").join("");
              return "";
            })
            .join("");
          json = { choices: [{ text: assembled }] };
        }
      } else {
        // try parse single-line JSON string
        try {
          json = JSON.parse(json);
        } catch (e) {
          // keep as string
        }
      }
    } else if (Array.isArray(json)) {
      // some servers may already return an array of fragment objects instead of NDJSON
      const fragments = json;
      if (fragments.length > 0) {
        const assembled = fragments
          .map((f: any) => {
            if (typeof f.response === "string") return f.response;
            if (typeof f.generated_text === "string") return f.generated_text;
            if (f.outputs && Array.isArray(f.outputs))
              return f.outputs.map((o: any) => o.text || o.content || "").join("");
            if (f.choices && Array.isArray(f.choices))
              return f.choices.map((c: any) => c.text || c.response || "").join("");
            return "";
          })
          .join("");
        json = { choices: [{ text: assembled }] };
      }
    }

    if (json && json.error) {
      throw new Error(json.error);
    }
    let numContentFiltered = 0;
    const completions = new Set<string>();
    if (this.isStarCoder) {
      completions.add(json.generated_text);
    } else {
      for (const choice of json.choices || [{ text: "" }]) {
        if (choice.finish_reason === "content_filter") {
          numContentFiltered++;
        }

        let text = choice.text?.trim() || "";
        const codeFenceRegex = /```[\s\S]*?```/g;
        const match = text.match(codeFenceRegex);
        if(match) {
          text = match.map((m: any) => m.replace(/```[\w-]*\n?/, "").replace(/```$/, "").trim()).join("\n\n");
        }
        completions.add(text);
      }
    }
    if (numContentFiltered > 0) {
      console.warn(
        `${numContentFiltered} completions were truncated due to content filtering.`
      );
    }
    return completions;
  }

  /**
   * Get completions from Codex and postprocess them as needed; print a warning if it did not produce any
   *
   * @param prompt the prompt to use
   */
  public async completions(
    prompt: string,
    temperature: number
  ): Promise<Set<string>> {
    try {
      let result = new Set<string>();
      for (const completion of await this.query(prompt, { temperature })) {
        result.add(trimCompletion(completion));
      }
      return result;
    } catch (err: any) {
      console.warn(`Failed to get completions: ${err.message}`);
      return new Set<string>();
    }
  }
}

if (require.main === module) {
  (async () => {
    const codex = new Codex(false);
    const prompt = fs.readFileSync(0, "utf8");
    const responses = await codex.query(prompt, { n: 1 });
    console.log([...responses][0]);
  })().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
