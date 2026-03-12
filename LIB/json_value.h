// json_value.h — Lightweight JSON parser & DOM for FlexAIDdS config files
//
// Zero external dependencies.  Supports the subset needed for config:
//   null, bool, int/double, string, array, object (string keys).
//
// Usage:
//   auto cfg = json::parse_file("config.json");
//   double T = cfg["thermodynamics"]["temperature"].as_double(300.0);
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <cctype>

namespace json {

class Value;
using Object = std::unordered_map<std::string, Value>;
using Array  = std::vector<Value>;

class Value {
public:
    enum Type { Null, Bool, Int, Double, String, ArrayT, ObjectT };

    Value() : data_(std::monostate{}) {}
    Value(bool v)               : data_(v) {}
    Value(int v)                : data_(static_cast<int64_t>(v)) {}
    Value(int64_t v)            : data_(v) {}
    Value(double v)             : data_(v) {}
    Value(const char* v)        : data_(std::string(v)) {}
    Value(std::string v)        : data_(std::move(v)) {}
    Value(Array v)              : data_(std::move(v)) {}
    Value(Object v)             : data_(std::move(v)) {}

    Type type() const {
        return static_cast<Type>(data_.index());
    }

    bool is_null()   const { return type() == Null; }
    bool is_bool()   const { return type() == Bool; }
    bool is_int()    const { return type() == Int; }
    bool is_double() const { return type() == Double; }
    bool is_number() const { return is_int() || is_double(); }
    bool is_string() const { return type() == String; }
    bool is_array()  const { return type() == ArrayT; }
    bool is_object() const { return type() == ObjectT; }

    // Accessors with defaults
    bool as_bool(bool fallback = false) const {
        if (is_bool()) return std::get<bool>(data_);
        return fallback;
    }
    int as_int(int fallback = 0) const {
        if (is_int()) return static_cast<int>(std::get<int64_t>(data_));
        if (is_double()) return static_cast<int>(std::get<double>(data_));
        return fallback;
    }
    double as_double(double fallback = 0.0) const {
        if (is_double()) return std::get<double>(data_);
        if (is_int()) return static_cast<double>(std::get<int64_t>(data_));
        return fallback;
    }
    float as_float(float fallback = 0.0f) const {
        return static_cast<float>(as_double(fallback));
    }
    unsigned int as_uint(unsigned int fallback = 0) const {
        return static_cast<unsigned int>(as_int(static_cast<int>(fallback)));
    }
    std::string as_string(const std::string& fallback = "") const {
        if (is_string()) return std::get<std::string>(data_);
        return fallback;
    }
    const Array& as_array() const {
        static const Array empty;
        if (is_array()) return std::get<Array>(data_);
        return empty;
    }
    const Object& as_object() const {
        static const Object empty;
        if (is_object()) return std::get<Object>(data_);
        return empty;
    }

    // Object lookup: returns Null value for missing keys
    const Value& operator[](const std::string& key) const {
        static const Value null_val;
        if (!is_object()) return null_val;
        auto& obj = std::get<Object>(data_);
        auto it = obj.find(key);
        return (it != obj.end()) ? it->second : null_val;
    }
    const Value& operator[](const char* key) const {
        return operator[](std::string(key));
    }

    // Array lookup
    const Value& operator[](size_t idx) const {
        static const Value null_val;
        if (!is_array()) return null_val;
        auto& arr = std::get<Array>(data_);
        return (idx < arr.size()) ? arr[idx] : null_val;
    }

    bool contains(const std::string& key) const {
        if (!is_object()) return false;
        return std::get<Object>(data_).count(key) > 0;
    }

    size_t size() const {
        if (is_array()) return std::get<Array>(data_).size();
        if (is_object()) return std::get<Object>(data_).size();
        return 0;
    }

    // Mutable object access (for building configs programmatically)
    Value& mut(const std::string& key) {
        if (!is_object()) data_ = Object{};
        return std::get<Object>(data_)[key];
    }

    void set(const std::string& key, Value v) {
        if (!is_object()) data_ = Object{};
        std::get<Object>(data_)[key] = std::move(v);
    }

private:
    std::variant<
        std::monostate,  // Null
        bool,            // Bool
        int64_t,         // Int
        double,          // Double
        std::string,     // String
        Array,           // Array
        Object           // Object
    > data_;
};

// ─── Recursive merge: b overrides a, objects merge recursively ────────────
inline Value merge(const Value& a, const Value& b) {
    if (a.is_object() && b.is_object()) {
        Object result = a.as_object();
        for (auto& [k, v] : b.as_object()) {
            auto it = result.find(k);
            if (it != result.end() && it->second.is_object() && v.is_object()) {
                result[k] = merge(it->second, v);
            } else {
                result[k] = v;
            }
        }
        return Value(std::move(result));
    }
    return b;
}

// ─── Minimal JSON parser ─────────────────────────────────────────────────

namespace detail {

class Parser {
public:
    explicit Parser(const std::string& src) : src_(src), pos_(0) {}

    Value parse() {
        skip_ws();
        Value v = parse_value();
        skip_ws();
        if (pos_ < src_.size())
            throw std::runtime_error("JSON: trailing content at position " + std::to_string(pos_));
        return v;
    }

private:
    const std::string& src_;
    size_t pos_;

    [[noreturn]] void error(const std::string& msg) {
        throw std::runtime_error("JSON parse error at " + std::to_string(pos_) + ": " + msg);
    }

    char peek() const {
        return pos_ < src_.size() ? src_[pos_] : '\0';
    }
    char advance() {
        if (pos_ >= src_.size()) error("unexpected end of input");
        return src_[pos_++];
    }
    void expect(char c) {
        char got = advance();
        if (got != c) {
            std::string msg = "expected '";
            msg += c;
            msg += "' but got '";
            msg += got;
            msg += "'";
            error(msg);
        }
    }
    void skip_ws() {
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else if (c == '/' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '/') {
                // Line comment (extension: helpful for config files)
                pos_ += 2;
                while (pos_ < src_.size() && src_[pos_] != '\n') ++pos_;
            } else {
                break;
            }
        }
    }

    Value parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string_value();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        error(std::string("unexpected character '") + c + "'");
    }

    Value parse_object() {
        expect('{');
        Object obj;
        skip_ws();
        if (peek() == '}') { advance(); return Value(std::move(obj)); }
        for (;;) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            skip_ws();
            obj[key] = parse_value();
            skip_ws();
            if (peek() == ',') { advance(); continue; }
            break;
        }
        skip_ws();
        expect('}');
        return Value(std::move(obj));
    }

    Value parse_array() {
        expect('[');
        Array arr;
        skip_ws();
        if (peek() == ']') { advance(); return Value(std::move(arr)); }
        for (;;) {
            skip_ws();
            arr.push_back(parse_value());
            skip_ws();
            if (peek() == ',') { advance(); continue; }
            break;
        }
        skip_ws();
        expect(']');
        return Value(std::move(arr));
    }

    std::string parse_string() {
        expect('"');
        std::string s;
        while (peek() != '"') {
            char c = advance();
            if (c == '\\') {
                char esc = advance();
                switch (esc) {
                    case '"':  s += '"'; break;
                    case '\\': s += '\\'; break;
                    case '/':  s += '/'; break;
                    case 'n':  s += '\n'; break;
                    case 't':  s += '\t'; break;
                    case 'r':  s += '\r'; break;
                    case 'b':  s += '\b'; break;
                    case 'f':  s += '\f'; break;
                    default:   s += esc; break;
                }
            } else {
                s += c;
            }
        }
        expect('"');
        return s;
    }

    Value parse_string_value() {
        return Value(parse_string());
    }

    Value parse_number() {
        size_t start = pos_;
        if (peek() == '-') advance();
        while (std::isdigit(static_cast<unsigned char>(peek()))) advance();
        bool is_float = false;
        if (peek() == '.') { is_float = true; advance(); while (std::isdigit(static_cast<unsigned char>(peek()))) advance(); }
        if (peek() == 'e' || peek() == 'E') { is_float = true; advance(); if (peek() == '+' || peek() == '-') advance(); while (std::isdigit(static_cast<unsigned char>(peek()))) advance(); }
        std::string num_str = src_.substr(start, pos_ - start);
        if (is_float) {
            return Value(std::stod(num_str));
        }
        int64_t iv = std::stoll(num_str);
        return Value(iv);
    }

    Value parse_bool() {
        if (src_.compare(pos_, 4, "true") == 0)  { pos_ += 4; return Value(true); }
        if (src_.compare(pos_, 5, "false") == 0) { pos_ += 5; return Value(false); }
        error("expected 'true' or 'false'");
    }

    Value parse_null() {
        if (src_.compare(pos_, 4, "null") == 0) { pos_ += 4; return Value(); }
        error("expected 'null'");
    }
};

} // namespace detail

// Parse a JSON string
inline Value parse(const std::string& json_str) {
    detail::Parser p(json_str);
    return p.parse();
}

// Parse a JSON file
inline Value parse_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open JSON file: " + path);
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return parse(ss.str());
}

} // namespace json
