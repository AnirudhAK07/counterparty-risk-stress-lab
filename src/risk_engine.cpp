// Counterparty Risk Stress Lab: deterministic FX-forward exposure calculator.
//
// Build: g++ -std=c++17 -O2 -Wall -Wextra -pedantic src/risk_engine.cpp -o risk_engine.exe
// Run:   risk_engine.exe trades.csv netting_sets.csv spot elapsed_days usd_rate eur_rate
//
// A positive signed EUR notional means the bank receives EUR and pays USD at
// maturity. A negative notional means the opposite. The spot quote is USD/EUR.
// Market value is calculated separately for each trade, then netted only within
// its contractual netting set. Received cash collateral reduces positive value.

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

struct NettingSet {
    std::string counterparty;
    double collateral_usd = 0.0;
    double mtm_usd = 0.0;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

// CSV names may contain a comma or a quote when the field is quoted. This
// parser deliberately rejects malformed quoting so bad input cannot silently
// shift the meaning of a column. Embedded newlines are outside this file format.
std::vector<std::string> parse_csv_line(const std::string& line,
                                        const std::string& location) {
    enum class State { unquoted, quoted, after_quote };
    State state = State::unquoted;
    std::vector<std::string> fields;
    std::string field;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (state == State::quoted) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    field += '"';
                    ++i;
                } else {
                    state = State::after_quote;
                }
            } else {
                field += c;
            }
        } else if (state == State::after_quote) {
            if (c != ',') fail(location + ": unexpected character after closing quote");
            fields.push_back(field);
            field.clear();
            state = State::unquoted;
        } else if (c == ',') {
            fields.push_back(field);
            field.clear();
        } else if (c == '"') {
            if (!field.empty()) fail(location + ": quote inside unquoted field");
            state = State::quoted;
        } else {
            field += c;
        }
    }
    if (state == State::quoted) fail(location + ": unterminated quoted field");
    fields.push_back(field);
    return fields;
}

std::vector<std::vector<std::string>> read_csv(
    const std::string& path, const std::vector<std::string>& required_header) {
    std::ifstream file(path);
    if (!file) fail("cannot open CSV file: " + path);

    std::vector<std::vector<std::string>> rows;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string location = path + ":" + std::to_string(line_number);
        if (line.empty()) fail(location + ": empty CSV line");
        auto fields = parse_csv_line(line, location);
        if (line_number == 1) {
            // A UTF-8 BOM is common in CSV files exported from spreadsheet apps.
            if (!fields.empty() && fields[0].compare(0, 3, "\xEF\xBB\xBF") == 0)
                fields[0].erase(0, 3);
            if (fields != required_header) fail(location + ": unexpected CSV header");
        } else {
            if (fields.size() != required_header.size())
                fail(location + ": wrong number of CSV columns");
            rows.push_back(std::move(fields));
        }
    }
    if (file.bad()) fail("error reading CSV file: " + path);
    if (line_number == 0) fail("empty CSV file: " + path);
    return rows;
}

double parse_double(const std::string& raw, const std::string& name) {
    if (raw.empty()) fail(name + ": expected a number");
    // from_chars is locale independent and requires the entire field to parse.
    const char* begin = raw.data();
    const char* end = begin + raw.size();
    if (*begin == '+') ++begin;
    if (begin == end) fail(name + ": expected a number");
    double value = 0.0;
    const auto result = std::from_chars(begin, end, value, std::chars_format::general);
    if (result.ec != std::errc{} || result.ptr != end || !std::isfinite(value))
        fail(name + ": expected a finite number, got '" + raw + "'");
    return value;
}

int parse_nonnegative_int(const std::string& raw, const std::string& name) {
    if (raw.empty()) fail(name + ": expected a nonnegative integer");
    int value = 0;
    const auto result = std::from_chars(raw.data(), raw.data() + raw.size(), value);
    if (result.ec != std::errc{} || result.ptr != raw.data() + raw.size() || value < 0)
        fail(name + ": expected a nonnegative integer, got '" + raw + "'");
    return value;
}

// Always quote text on output; this keeps counterparty and set IDs safe when
// they contain punctuation. Numeric output is fixed to a reproducible precision.
std::string csv_text(const std::string& input) {
    std::string escaped = "\"";
    for (char c : input) {
        if (c == '"') escaped += '"';
        escaped += c;
    }
    escaped += '"';
    return escaped;
}

void require_identifier(const std::string& value, const std::string& name) {
    if (value.empty()) fail(name + ": identifier cannot be empty");
    if (value.find('\n') != std::string::npos || value.find('\r') != std::string::npos)
        fail(name + ": identifier cannot contain a newline");
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc != 7) {
            fail("usage: risk_engine <trades.csv> <netting_sets.csv> "
                 "<spot_usd_per_eur> <elapsed_calendar_days> <usd_rate> <eur_rate>");
        }
        const double spot = parse_double(argv[3], "spot_usd_per_eur");
        const int elapsed_days = parse_nonnegative_int(argv[4], "elapsed_calendar_days");
        const double usd_rate = parse_double(argv[5], "usd_rate");
        const double eur_rate = parse_double(argv[6], "eur_rate");
        if (spot <= 0.0) fail("spot_usd_per_eur must be positive");

        const auto set_rows = read_csv(argv[2],
            {"netting_set", "counterparty", "collateral_usd"});
        std::map<std::string, NettingSet> sets;
        for (std::size_t i = 0; i < set_rows.size(); ++i) {
            const auto& row = set_rows[i];
            const std::string where = std::string(argv[2]) + ":" + std::to_string(i + 2);
            require_identifier(row[0], where + " netting_set");
            require_identifier(row[1], where + " counterparty");
            const double collateral = parse_double(row[2], where + " collateral_usd");
            if (collateral < 0.0) fail(where + ": collateral_usd must be nonnegative");
            if (!sets.emplace(row[0], NettingSet{row[1], collateral, 0.0}).second)
                fail(where + ": duplicate netting_set '" + row[0] + "'");
        }
        if (sets.empty()) fail("netting_sets.csv has no data rows");

        const auto trade_rows = read_csv(argv[1],
            {"trade_id", "netting_set", "signed_eur_notional", "strike", "maturity_days"});
        std::set<std::string> trade_ids;
        for (std::size_t i = 0; i < trade_rows.size(); ++i) {
            const auto& row = trade_rows[i];
            const std::string where = std::string(argv[1]) + ":" + std::to_string(i + 2);
            require_identifier(row[0], where + " trade_id");
            require_identifier(row[1], where + " netting_set");
            if (!trade_ids.insert(row[0]).second)
                fail(where + ": duplicate trade_id '" + row[0] + "'");
            auto set_it = sets.find(row[1]);
            if (set_it == sets.end())
                fail(where + ": unknown netting_set '" + row[1] + "'");
            const double notional = parse_double(row[2], where + " signed_eur_notional");
            const double strike = parse_double(row[3], where + " strike");
            const int maturity_days = parse_nonnegative_int(row[4], where + " maturity_days");
            if (strike <= 0.0) fail(where + ": strike must be positive");
            if (maturity_days <= elapsed_days)
                fail(where + ": trade has matured at this elapsed day");

            // Remaining time to maturity uses an ACT/365 approximation.
            const double tau = static_cast<double>(maturity_days - elapsed_days) / 365.0;
            const double eur_leg = spot * std::exp(-eur_rate * tau);
            const double usd_leg = strike * std::exp(-usd_rate * tau);
            const double mtm = notional * (eur_leg - usd_leg);
            if (!std::isfinite(eur_leg) || !std::isfinite(usd_leg) || !std::isfinite(mtm))
                fail(where + ": valuation overflowed; check rates and trade size");
            set_it->second.mtm_usd += mtm;
            if (!std::isfinite(set_it->second.mtm_usd))
                fail(where + ": netting-set value overflowed");
        }
        if (trade_rows.empty()) fail("trades.csv has no data rows");

        struct OutputRow {
            std::string counterparty;
            std::string set_id;
            double mtm;
            double collateral;
            double exposure;
        };
        std::vector<OutputRow> output;
        for (const auto& entry : sets) {
            const auto& set = entry.second;
            // Netting never crosses set boundaries, even for one counterparty.
            const double exposure = std::max(set.mtm_usd - set.collateral_usd, 0.0);
            if (!std::isfinite(exposure)) fail("exposure overflowed for " + entry.first);
            output.push_back({set.counterparty, entry.first, set.mtm_usd,
                              set.collateral_usd, exposure});
        }
        std::sort(output.begin(), output.end(), [](const auto& a, const auto& b) {
            if (a.counterparty != b.counterparty) return a.counterparty < b.counterparty;
            return a.set_id < b.set_id;
        });

        std::cout << "counterparty,netting_set,mtm_usd,collateral_usd,exposure_usd\n";
        std::cout << std::fixed << std::setprecision(6);
        for (const auto& row : output) {
            std::cout << csv_text(row.counterparty) << ',' << csv_text(row.set_id) << ','
                      << row.mtm << ',' << row.collateral << ',' << row.exposure << '\n';
        }
        if (!std::cout) fail("error writing result to standard output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "risk_engine: " << error.what() << '\n';
        return 1;
    }
}
