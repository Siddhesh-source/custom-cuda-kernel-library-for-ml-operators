#include "bench/support/shapes.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ccml_bench {

namespace {

/* A deliberately small recursive-descent reader for the subset of JSON the
   shape file uses: objects, arrays, strings without escapes beyond \" and \\,
   integers, true/false/null. Keeps the benchmarks free of a JSON dependency. */
class Reader {
 public:
  explicit Reader(const std::string& text) : s_(text) {}

  void skip() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) {
      ++i_;
    }
  }
  bool peek(char c) {
    skip();
    return i_ < s_.size() && s_[i_] == c;
  }
  void expect(char c) {
    skip();
    if (i_ >= s_.size() || s_[i_] != c) {
      fail(std::string("expected '") + c + "'");
    }
    ++i_;
  }
  std::string string_value() {
    expect('"');
    std::string out;
    while (i_ < s_.size() && s_[i_] != '"') {
      if (s_[i_] == '\\' && i_ + 1 < s_.size()) {
        ++i_;
      }
      out += s_[i_++];
    }
    expect('"');
    return out;
  }
  long long integer_value() {
    skip();
    const char* begin = s_.c_str() + i_;
    char* end = nullptr;
    const long long v = std::strtoll(begin, &end, 10);
    if (end == begin) {
      fail("expected an integer");
    }
    i_ += static_cast<std::size_t>(end - begin);
    return v;
  }
  /* Skips any value; used for keys the reader does not know about. */
  void skip_value() {
    skip();
    if (peek('"')) {
      string_value();
    } else if (peek('{')) {
      expect('{');
      if (!peek('}')) {
        do {
          string_value();
          expect(':');
          skip_value();
        } while (consume(','));
      }
      expect('}');
    } else if (peek('[')) {
      expect('[');
      if (!peek(']')) {
        do {
          skip_value();
        } while (consume(','));
      }
      expect(']');
    } else {
      while (i_ < s_.size() && std::string(",}] \t\r\n").find(s_[i_]) == std::string::npos) {
        ++i_;
      }
    }
  }
  bool consume(char c) {
    if (peek(c)) {
      ++i_;
      return true;
    }
    return false;
  }
  [[noreturn]] void fail(const std::string& what) const {
    throw std::runtime_error("shapes.json: " + what + " at offset " +
                             std::to_string(i_));
  }

 private:
  const std::string& s_;
  std::size_t i_ = 0;
};

ShapeEntry read_entry(Reader& r) {
  ShapeEntry e;
  r.expect('{');
  if (!r.peek('}')) {
    do {
      const std::string key = r.string_value();
      r.expect(':');
      if (key == "tag") e.tag = r.string_value();
      else if (key == "M") e.M = r.integer_value();
      else if (key == "N") e.N = r.integer_value();
      else if (key == "K") e.K = r.integer_value();
      else if (key == "rows") e.rows = r.integer_value();
      else if (key == "cols") e.cols = r.integer_value();
      else r.skip_value();
    } while (r.consume(','));
  }
  r.expect('}');
  return e;
}

}  // namespace

std::vector<ShapeEntry> parse_shapes(const std::string& json, const std::string& op) {
  Reader r(json);
  std::vector<ShapeEntry> out;
  bool found = false;
  r.expect('{');
  if (!r.peek('}')) {
    do {
      const std::string key = r.string_value();
      r.expect(':');
      if (key == op) {
        found = true;
        r.expect('[');
        if (!r.peek(']')) {
          do {
            out.push_back(read_entry(r));
          } while (r.consume(','));
        }
        r.expect(']');
      } else {
        r.skip_value();
      }
    } while (r.consume(','));
  }
  r.expect('}');
  if (!found) {
    throw std::runtime_error("shapes.json: no \"" + op + "\" section");
  }
  return out;
}

std::vector<ShapeEntry> load_shapes(const std::string& path, const std::string& op) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("cannot open " + path);
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return parse_shapes(buffer.str(), op);
}

}  // namespace ccml_bench
