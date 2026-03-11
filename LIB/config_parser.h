#pragma once

#include <string>
#include <nlohmann/json.hpp>

// Forward declarations
struct FA_Global_struct;
typedef struct FA_Global_struct FA_Global;
struct GB_Global_struct;
typedef struct GB_Global_struct GB_Global;

// Load a JSON config file and merge it on top of defaults.
// Returns the merged config (defaults + user overrides).
nlohmann::json load_config(const std::string& config_path);

// Apply merged JSON config to FA_Global and GB_Global structs.
// This maps JSON keys to the existing C struct fields.
void apply_config(const nlohmann::json& config, FA_Global* FA, GB_Global* GB);

// Apply --rigid overrides: disables all flexibility, sets temperature to 0.
nlohmann::json rigid_overrides();

// Merge two JSON objects (b overrides a, recursively for objects).
nlohmann::json merge_json(const nlohmann::json& a, const nlohmann::json& b);
