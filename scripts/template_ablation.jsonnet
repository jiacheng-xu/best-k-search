// A regular definition.
local task_name="cg";
// local task_name="squad";
// local task_name="drop";
// local task_name="quoref";
// local task_name="gec";
// local task_name="opus";
// local task_name="xsum-bart";

local use_debug = "--debug";
// local use_debug = "";

// local use_constraint = "--constraint";
local use_constraint = "";


local typ_seed = ["0.2", "0.5", "0.95" ];
local top_seed = ["0.5", "0.8", "0.9" ];



local beam_size = if task_name=="opus" then 8 else 10;
local grp_size = if beam_size==10 then ["5","10"] else ["4","8"];
local heap_top=if beam_size==10 then [ "10", "15"] else [ "8" , "10"];

// pairs for complete hyper-param sweep
local pairs = [ ["--group_size", "5", "--heap_sample", "--heap_top_k","5"], ["--group_size", "5", "--heap_sample", "--heap_top_k","8"],["--group_size", "5", "--heap_sample", "--heap_top_k","10"], ["--group_size", "10", "--heap_sample", "--heap_top_k","10"], ["--group_size", "10", "--heap_sample", "--heap_top_k","15"],["--group_size", "10", "--heap_sample", "--heap_top_k","20"],];

// deterministic pairs
local pairs = [ ["--group_size", "5", "--heap_sample", "--heap_top_k","5"], ["--group_size", "10", "--heap_sample", "--heap_top_k","10"]];

local config_squad=["--max_len", "25","--split","validation"];
local config_drop=["--max_len", "20","--split","validation"];
local config_quoref=["--max_len", "25","--split","validation"];
local config_cg=["--max_len", "20"];
local config_gec=["--max_len", "-1"];
local config_opus=["--max_len", "-1"];
local config_xsum=["--max_len", "30","--split","test"];


local bs = ["--algo", "bs"];
local dbs = ["--algo", "bs", "--diversity_penalty","5", "--num_beam_groups", grp_size[0]];   # relevant to beam size
local dbs_full = ["--algo", "bs","--diversity_penalty","5", "--num_beam_groups", grp_size[1] ];   # relevant to beam size
// local bfs =["--algo", "bfs"];
local sample = {
  local typ_p = typ_seed,
  local top_p = top_seed,
  local common = ["--algo", "sample"],
  "typical_p": [ ["--typical_p", x] + common   for x in typ_p],
  "top_p": [ ["--top_p", x] + common  for x in top_p],
};
local sample1 = {
  local typ_p = typ_seed,
  local top_p = top_seed,
  local common = ["--algo", "sample1"],
  "typical_p": [ ["--typical_p", x] + common   for x in typ_p],
  "top_p": [ ["--top_p", x] + common  for x in top_p],
};

local FuncBfs() = {
  local prefix = ["--algo", "bfs"],
  local tmp=["0.0", "0.01","0.05", "0.1", "0.2"],
  local len_fact=["1","-1"],
  // "args": [ prefix + ["--group_size", g,"--temp_decay", t, "--heap_sample", "--heap_top_k", htop] for htop in heap_top for t in tmp for g in grp_size],
  "args": [prefix + p + [ "--temp_decay", t, "--len_factor", lf] for lf in len_fact for t in tmp for p in pairs ],
};
local bfs = FuncBfs()["args"];

local _algos = bfs ;
// local _algos = [ dbs, dbs_full ];

local algos = [ ["--beam_size", beam_size ] + x for x in _algos];

local TaskConfig(task) = {
  local config= if task=="squad" then config_squad else 
        if task == 'quoref' then config_quoref else
        if task == 'drop' then config_drop else
        if task == "cg" then config_cg else 
        if task == "gec" then config_gec else
        if task == "opus" then config_opus else
        if task == "xsum"  then config_xsum else
        if task == "xsum-peg" then config_xsum else
        if task == "xsum-bart" then config_xsum else
        config_quoref,
  "config":config
};
local task_conf =  TaskConfig(task_name)["config"];

local base={            
            "name": "Python: QG",
            "type": "python",
            "request": "launch",
            "program": "bbfs/run.py",
            "console": "integratedTerminal",
            "justMyCode": true,};
{
  // "sample":sample,
  // "al":algos
  "args": [ x + task_conf + ["--task",task_name, use_debug, use_constraint] for x in algos],
}