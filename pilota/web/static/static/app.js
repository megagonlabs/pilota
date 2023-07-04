const QUERY_PREFIX = "query_";
const API_PREFIX = "";

function set_location_hash(app) {
  const GIVEN_PARAMS = {
    pilota: ["context", "user"],
  }[app.appname];

  const kv = {};
  GIVEN_PARAMS.forEach((p) => {
    const v = app[`${QUERY_PREFIX}${p}`];
    kv[p] = v;
  });
  location.hash = LZString.compressToEncodedURIComponent(JSON.stringify(kv));
}

async function set_template(app) {
  if (app.appname == "pilota") {
    await axios.get("complement.json").then((response) => {
      const cc = document.getElementById("context_candidates");
      response.data.template.forEach((v) => {
        const btn = document.createElement("button");
        btn.className = "btn btn-outline-primary form-control mb-3";
        btn.style = "text-align:left;";
        btn.innerText = v;
        btn.value = v;
        btn.addEventListener("click", (ev) => {
          app["query_context"] = v;
          document.getElementById("exampleModalClose").click();
        });
        cc.appendChild(btn);
      });
    });
  }
}

function set_default(app) {
  if (app.appname == "pilota") {
    app["query_context"] = "ご要望をお知らせください";
    app["query_user"] =
      "はい。部屋から富士山が見えて、夜景を見ながら食事のできるホテルがいいな。";
    return;
  }
  throw "Unknown appname";
}

function get_query(app) {
  const query = {};
  if (app.appname == "pilota") {
    const texts = app[`${QUERY_PREFIX}context`].split("\n//\n");

    if (app["use_context"]) {
      query["context"] = [];
      for (let j = 0; j < texts.length; ++j) {
        query["context"].push({
          name: (texts.length - j) % 2 == 0 ? "user" : "agent",
          text: texts[j].trim(),
        });
      }
    } else {
      query["context"] = null;
    }
    query["utterance"] = app[`${QUERY_PREFIX}user`].trim();

    query["param"] = {};
    app.param_keys.forEach((key) => {
      query["param"][key] = app[`${QUERY_PREFIX}param_${key}`];
    });
    query["model_name"] = app[`${QUERY_PREFIX}model_name`];
    query["only_best"] = false;
  } else {
    throw "Unknown appname";
  }

  app.query = query;
  return query;
}

function show_message(text) {
  const mb = document.getElementById("info_message");
  if (mb) {
    mb.style.visibility = "visible";
    mb.innerText = text;
    setTimeout(function () {
      mb.style.visibility = "hidden";
    }, 1000);
  }
}

const app = new Vue({
  el: "#app",
  data: {
    resp: {},
    query: undefined,
    error: false,
    param_keys: undefined,

    query_context: "",
    query_user: "",
    query_param_in_max_length: 0,
    query_param_out_max_length: 0,
    query_param_beam_search: 1,
    query_param_nbest: 1,
    query_param_repetition_penalty: 2.5,
    query_param_length_penalty: 1,
    query_param_temperature: 1.0,
    query_param_diversity_penalty: 0.0,
    query_param_num_beam_groups: 1,
    query_param_early_stopping: true,
    query_param_top_k: 0,
    query_param_top_p: 0,
    query_param_typical_p: 0,
    query_param_rerank: false,
    query_param_force_gold: true,

    query_text: "",
    query_param_number: 10,
    query_param_threshold: 0.5,
    query_param_k: 50,
    query_param_force_gold: true,
    query_param_level_min: 0,
    query_param_level_max: 99,
    query_model_name: "UNK",
    query_area_cds: "",
    model2config: undefined,
    query_rte_prompt_template: "",
    rte_prompt_templates: [
      "という要望に適切なトピックを、適切な順に理由・確率とともに列挙してください。",
    ],

    url_api: undefined,
    appname: undefined,
    keyDownCode: 0,

    elapsed_time: 0,
  },

  computed: {
    use_context: function () {
      if (this.model2config === undefined) {
        return true;
      }
      const use =
        this.model2config[this.query_model_name]["acceptable_names"].length > 1;
      if (!use) {
        this.query_context = ""; // Clear context
      }
      return use;
    },
  },

  methods: {
    // https://qiita.com/TK-C/items/4b32a3f98343606d979f
    keyDownEnter(e) {
      this.keyDownCode = e.keyCode; //enterを押した時のkeycodeを記録
      e.preventDefault();
    },
    keyUpEnter(e) {
      if (this.keyDownCode === 229) {
        //229コードの場合は処理をストップ
        return;
      }
      e.preventDefault();
      this.send();
    },
    keyEnterShift(e) {
      console.log("shift,enter");
    },
    send() {
      document.getElementById("button_search").click();
    },

    search: async function () {
      const btn = document.getElementById("button_search");
      {
        btn.disabled = true;
        btn.textContent = "処理中";
        this.error = false;
        this.resp = {};
        document.getElementById("info").style.display = "none";
        document.getElementById("jsondump").style.display = "none";
        document.getElementById("loader").style.display = "block";
        set_location_hash(this);
      }

      try {
        const queries = get_query(this);
        const startTime = performance.now();
        await axios
          .post(this.url_api, queries)
          .then((response) => {
            this.resp = Object.assign({}, this.resp, response.data);
          })
          .catch((error) => {
            this.error = true;
            this.resp = Object.assign({}, this.resp, error.response);
          });

        if (this.resp.scuds) {
          for (let j = 0; j < this.resp.scuds.length; ++j) {
            if (this.resp.scuds[j].length == 0) {
              this.resp.scuds[j] = ["<none>"];
            }
          }
        }

        const endTime = performance.now();
        this.elapsed_time = (endTime - startTime) / 1000;
      } catch (e) {}

      {
        btn.textContent = "送信";
        btn.disabled = false;
        document.getElementById("info").style.display = "block";
        document.getElementById("jsondump").style.display = "block";
        document.getElementById("loader").style.display = "none";
      }
    },

    set_model_names: async function () {
      const select = document.getElementById("query_model_name");
      if (select === null) {
        return;
      }

      try {
        const url_info = `${API_PREFIX}api/info`;
        const response = await axios.get(url_info);
        this.resp = Object.assign({}, this.resp, response.data);
        this.query = { url: url_info };

        // set default model names
        {
          const mns = response.data.models;
          mns.push("default");
          for (let mn of mns) {
            const opt = document.createElement("option");
            opt.text = mn;
            opt.value = mn;
            select.appendChild(opt);
          }
          this.query_model_name = "default"; //response.data.default;
        }
        if (response.data.configs !== undefined) {
          this.model2config = response.data.configs;
          this.model2config["default"] =
            response.data.configs[response.data.default];
        }

        // set default parameters
        {
          const dp = response.data.param;
          app.param_keys = Object.keys(dp);

          for (let k in dp) {
            const v = dp[k];
            this[`${QUERY_PREFIX}param_${k}`] = v;
          }

          // Forced default
          this[`${QUERY_PREFIX}param_nbest`] = 5;
          this[`${QUERY_PREFIX}param_beam_search`] = 5;
          this[`${QUERY_PREFIX}param_score`] = true;
        }
      } catch (error) {
        this.error = Object.assign({}, this.error, error);
      }
    },
  },
});

document.addEventListener("DOMContentLoaded", async function () {
  const scud_btn = document.getElementById("button_scud");
  if (scud_btn) {
    scud_btn.addEventListener("click", () => {
      let ret = [];
      app.resp.scuds.forEach((v) => {
        ret = ret.concat(v);
      });
      navigator.clipboard.writeText(ret.join("\n"));
      show_message("Copied!");
    });
  }

  const all_btn = document.getElementById("button_all");
  if (all_btn) {
    all_btn.addEventListener("click", () => {
      let ret = [
        `Context: ${app.query.context[0].text}`,
        `Param: ${JSON.stringify(app.query.param)}`,
        `Model: ${app.resp.model_name}`,
        "",
      ];
      const nbest = app.resp.scuds.length / app.resp.sentences.length;
      for (let j = 0; j < app.resp.sentences.length; ++j) {
        if (j != 0) {
          ret = ret.concat("");
        }
        ret = ret.concat(`# ${app.resp.sentences[j]}`);
        for (let k = 0; k < nbest; ++k) {
          const scuds = app.resp.scuds[j * nbest + k];
          if (nbest > 1) {
            ret = ret.concat(`${k + 1}:`);
          }
          ret = ret.concat(scuds.join("\n"));
        }
      }

      navigator.clipboard.writeText(ret.join("\n"));
      show_message("Copied!");
    });
  }

  const share_btn = document.getElementById("button_share");
  if (share_btn) {
    share_btn.addEventListener("click", () => {
      navigator.clipboard.writeText(location.href);
      show_message("Copied!");
    });
  }

  app.appname = document.getElementById("app").dataset.appname;
  app.url_api = {
    pilota: `${API_PREFIX}api/predict`,
  }[app.appname];

  await app.set_model_names();
  await set_template(app);

  if (location.hash.length == 0) {
    set_default(app);
    return;
  }

  try {
    const kv = JSON.parse(
      LZString.decompressFromEncodedURIComponent(location.hash.substring(1))
    );

    for (const _k in kv) {
      app[`${QUERY_PREFIX}${_k}`] = kv[_k];
    }
    document.getElementById("button_search").click();
  } catch (error) {
    app.error = Object.assign({}, app.error, {
      message: `Error: Invalid parameters\n${error}`,
    });
  }
});
