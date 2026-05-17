#include "go_game.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <unordered_set>

namespace alpha_go {

namespace {

BoardTables build_board_tables(int size) {
    BoardTables t;
    const int n = size * size;
    t.neighbor_indices.resize(n);
    t.neighbor_counts.resize(n);
    t.zobrist.resize(n);

    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            int idx = row * size + col;
            int count = 0;
            if (row > 0)         t.neighbor_indices[idx][count++] = (row - 1) * size + col;
            if (row < size - 1)  t.neighbor_indices[idx][count++] = (row + 1) * size + col;
            if (col > 0)         t.neighbor_indices[idx][count++] = row * size + (col - 1);
            if (col < size - 1)  t.neighbor_indices[idx][count++] = row * size + (col + 1);
            t.neighbor_counts[idx] = count;
        }
    }

    // Deterministic seed so two boards of the same size hash identically.
    // (Different seeds per size since N² varies.) splitmix64 gives
    // well-distributed 64-bit values.
    uint64_t seed = 0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(size);
    auto next = [&seed]() -> uint64_t {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    for (int i = 0; i < n; ++i) {
        t.zobrist[i][GoBoard::EMPTY] = 0;       // empty contributes nothing
        t.zobrist[i][GoBoard::BLACK] = next();
        t.zobrist[i][GoBoard::WHITE] = next();
    }
    return t;
}

}  // namespace

const BoardTables& board_tables(int size) {
    // Built once per size, then read-only for the program lifetime.
    // GoBoard construction is the only caller (never per-copy), so the
    // mutex is uncontended in practice; it only guards the rare
    // first-touch insert against the leaf-parallel batched path.
    static std::mutex mu;
    static std::map<int, std::unique_ptr<BoardTables>> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(size);
    if (it == cache.end()) {
        it = cache.emplace(size,
                            std::make_unique<BoardTables>(build_board_tables(size)))
                 .first;
    }
    return *it->second;
}

GoBoard::GoBoard(int size, float komi)
    : size_(size),
      komi_(komi),
      board_(size * size, EMPTY),
      tables_(&board_tables(size)) {
    // Empty board hash = 0 (XOR of zero stones). seen_hashes_ starts
    // empty — the first move's resulting hash is unconditionally legal
    // (no prior states to repeat), and the empty-board hash gets added
    // to seen_hashes_ when the first move is played.
}

std::pair<std::vector<int>, std::vector<int>> GoBoard::get_group_and_liberties(int index) const {
    int8_t color = board_[index];
    if (color == EMPTY) {
        return {{}, {}};
    }

    std::vector<int> group;
    std::vector<int> liberties;
    std::vector<bool> visited(size_ * size_, false);
    std::vector<bool> liberty_visited(size_ * size_, false);

    std::queue<int> queue;
    queue.push(index);
    visited[index] = true;

    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();
        group.push_back(current);

        for (int i = 0; i < tables_->neighbor_counts[current]; ++i) {
            int neighbor = tables_->neighbor_indices[current][i];

            if (board_[neighbor] == EMPTY) {
                if (!liberty_visited[neighbor]) {
                    liberty_visited[neighbor] = true;
                    liberties.push_back(neighbor);
                }
            } else if (board_[neighbor] == color && !visited[neighbor]) {
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }

    return {group, liberties};
}

int GoBoard::remove_group(const std::vector<int>& group) {
    for (int idx : group) {
        // XOR-out the stone's contribution from current_hash_ before
        // erasing it. Symmetric to the placement XOR in play_flat_unchecked.
        current_hash_ ^= tables_->zobrist[idx][board_[idx]];
        board_[idx] = EMPTY;
    }
    return static_cast<int>(group.size());
}

bool GoBoard::is_legal(int row, int col) const {
    return is_legal_flat(flat_index(row, col));
}

bool GoBoard::is_legal_flat(int index) const {
    // Tromp-Taylor legality, with two practical departures from full TT to
    // keep this engine in lock-step with KataGo (which is what we play
    // games against):
    //
    //   1. We use the simple single-stone ko rule + positional superko
    //      (PSK) layer for repetitions beyond the immediate ko. KataGo
    //      can be configured for SIMPLE ko via `koRule = SIMPLE` in
    //      gtp_example.cfg; PSK never fires on the standard simple-ko
    //      patterns, only on rarer board-state repetitions.
    //
    //   2. **All suicide is illegal here**, both single-stone and
    //      multi-stone. KataGo's default rule profile is `suicide:false`
    //      (i.e. `multiStoneSuicideLegal = false`), so any suicide we
    //      accept that KataGo silently rejects desyncs the two
    //      state-trackers. We previously allowed multi-stone suicide,
    //      which caused mid-game illegal-move crashes once the white
    //      agent placed an inside-its-own-group stone (C++ self-captured
    //      and removed the white group; KataGo refused the move; the
    //      boards diverged from there). Rejecting all suicide
    //      universally is the simplest alignment.
    //
    // A placement at `index` is suicide iff, after capturing any
    // opponent groups whose only liberty is `index`, the resulting
    // friendly group has zero liberties. Detected by simulating the
    // move on a temporary copy.
    if (index < 0 || index >= size_ * size_) {
        return false;
    }
    if (board_[index] != EMPTY) {
        return false;
    }
    if (ko_point_.has_value() && ko_point_.value() == index) {
        return false;
    }
    int8_t player = to_play_;
    int8_t opponent = (player == BLACK) ? WHITE : BLACK;
    bool has_friendly_neighbor = false;
    bool has_empty_neighbor = false;
    bool captures_opponent = false;
    for (int i = 0; i < tables_->neighbor_counts[index]; ++i) {
        int neighbor = tables_->neighbor_indices[index][i];
        int8_t v = board_[neighbor];
        if (v == EMPTY) {
            has_empty_neighbor = true;
            break;  // immediate liberty — never suicide
        } else if (v == player) {
            has_friendly_neighbor = true;
        } else if (v == opponent && !captures_opponent) {
            // Would our placement capture this opponent group? It does
            // iff the group's only remaining liberty is `index` itself.
            auto [_group, liberties] = get_group_and_liberties(neighbor);
            if (liberties.size() == 1 && liberties[0] == index) {
                captures_opponent = true;
            }
        }
    }
    // Single-stone suicide fast-reject: no friendly neighbors AND no
    // immediate liberty AND no capture. The placement is a lone stone
    // surrounded by opponent stones that won't die — clearly suicide,
    // and we can reject without paying for a simulation.
    if (!has_empty_neighbor && !has_friendly_neighbor && !captures_opponent) {
        return false;
    }

    // Simulate the move once to check (a) multi-stone suicide and (b)
    // positional superko. Both inspections need the post-move board
    // state, so they share the simulation.
    bool need_suicide_check = !has_empty_neighbor && !captures_opponent;
    bool need_psk_check = !seen_hashes_.empty();
    if (need_suicide_check || need_psk_check) {
        // Lightweight clone: skips copy-constructing (then discarding)
        // the whole seen_hashes_ set. We only need the resulting hash +
        // occupancy at `index`; PSK is checked against *this's history.
        GoBoard tmp(*this, SimClone{});
        tmp.play_flat_unchecked(index);
        // (a) Multi-stone suicide: play_flat_unchecked's self-capture
        // branch removes our group when it has no liberties post-move,
        // leaving `index` empty. That's how we detect it post-hoc.
        if (need_suicide_check && tmp.board_[index] == EMPTY) {
            return false;
        }
        // (b) Positional superko: the resulting state matches a prior
        // board we've already reached.
        if (need_psk_check
                && seen_hashes_.find(tmp.current_hash_) != seen_hashes_.end()) {
            return false;
        }
    }
    return true;
}

std::vector<int> GoBoard::get_legal_moves_flat() const {
    std::vector<int> moves;
    moves.reserve(size_ * size_);

    for (int i = 0; i < size_ * size_; ++i) {
        if (is_legal_flat(i)) {
            moves.push_back(i);
        }
    }

    return moves;
}

bool GoBoard::play(int row, int col) {
    return play_flat(flat_index(row, col));
}

bool GoBoard::play_flat(int index) {
    if (!is_legal_flat(index)) {
        return false;
    }
    play_flat_unchecked(index);
    return true;
}

void GoBoard::play_flat_unchecked(int index) {
    // Insert the pre-move board hash into seen_hashes_ — this is the
    // state we're moving AWAY from. After this call, current_hash_ is
    // updated to the post-move state, which is NOT yet in seen_hashes_
    // (it gets added on the NEXT move). That ordering is what makes
    // pass legal under PSK: passing produces a board equal to the
    // current one, whose hash isn't in seen_hashes_ until pass() runs.
    seen_hashes_.insert(current_hash_);

    // Place the stone (and update the running hash).
    board_[index] = to_play_;
    current_hash_ ^= tables_->zobrist[index][to_play_];
    int8_t opponent = (to_play_ == BLACK) ? WHITE : BLACK;

    // Reset ko point
    ko_point_ = std::nullopt;

    // Check for captures
    int total_captured = 0;
    int last_captured_idx = -1;

    for (int i = 0; i < tables_->neighbor_counts[index]; ++i) {
        int neighbor = tables_->neighbor_indices[index][i];
        if (board_[neighbor] == opponent) {
            auto [group, liberties] = get_group_and_liberties(neighbor);
            if (liberties.empty()) {
                // Capture this group
                if (group.size() == 1) {
                    last_captured_idx = group[0];
                }
                total_captured += remove_group(group);
            }
        }
    }

    // Self-capture: after opponent groups are removed, if our own group
    // is left with no liberties, remove it. ALL suicide (single-stone
    // and multi-stone) is filtered out by is_legal_flat to stay aligned
    // with KataGo's default `suicide:false` rule. This branch is kept
    // because is_legal_flat itself invokes play_flat_unchecked on a
    // temporary copy to *detect* multi-stone suicide (post-call check:
    // is `index` empty?) — so this self-capture logic is what makes
    // that detection work. It should not fire on the real game-state
    // play_flat path because is_legal_flat rejects suicide upstream.
    //
    // The classic single-stone ko rule still applies on the more common
    // path where the move captured exactly one opponent stone and our
    // placement is itself a singleton with one liberty.
    auto [our_group, our_liberties] = get_group_and_liberties(index);
    if (our_liberties.empty()) {
        remove_group(our_group);
    } else if (total_captured == 1 && our_group.size() == 1 && our_liberties.size() == 1) {
        ko_point_ = last_captured_idx;
    }

    // Reset consecutive-pass streak — any non-pass move ends a pass run.
    consecutive_passes_ = 0;
    move_count_++;

    // Switch player
    to_play_ = opponent;
}

bool GoBoard::pass() {
    // PSK bookkeeping: passing leaves the board unchanged (current_hash_
    // doesn't change), but the *current* state has now been "visited"
    // and should be in seen_hashes_ so future moves can't return to it.
    // Adding to a set is idempotent on duplicates, so this is safe even
    // if the same hash later gets inserted by play_flat_unchecked.
    seen_hashes_.insert(current_hash_);
    consecutive_passes_++;
    move_count_++;
    to_play_ = (to_play_ == BLACK) ? WHITE : BLACK;
    ko_point_ = std::nullopt;
    return true;
}

float GoBoard::score() const {
    // Tromp-Taylor area scoring: all stones counted as alive, empty regions
    // awarded to a color iff they touch only that color.
    float black_score = 0.0f;
    float white_score = komi_;

    // Count stones
    for (int i = 0; i < size_ * size_; ++i) {
        if (board_[i] == BLACK) {
            black_score += 1.0f;
        } else if (board_[i] == WHITE) {
            white_score += 1.0f;
        }
    }

    // Count territory (empty regions surrounded by one color)
    std::vector<bool> visited(size_ * size_, false);

    for (int i = 0; i < size_ * size_; ++i) {
        if (board_[i] != EMPTY || visited[i]) {
            continue;
        }

        // Flood-fill to find empty region
        std::vector<int> territory;
        std::unordered_set<int8_t> borders;
        std::queue<int> queue;
        queue.push(i);
        visited[i] = true;

        while (!queue.empty()) {
            int current = queue.front();
            queue.pop();
            territory.push_back(current);

            for (int j = 0; j < tables_->neighbor_counts[current]; ++j) {
                int neighbor = tables_->neighbor_indices[current][j];
                if (board_[neighbor] == EMPTY) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        queue.push(neighbor);
                    }
                } else {
                    borders.insert(board_[neighbor]);
                }
            }
        }

        // If territory is surrounded by only one color, count it
        if (borders.size() == 1) {
            float territory_size = static_cast<float>(territory.size());
            if (*borders.begin() == BLACK) {
                black_score += territory_size;
            } else {
                white_score += territory_size;
            }
        }
    }

    return black_score - white_score;
}

int8_t GoBoard::get_winner() const {
    float s = score();
    if (s > 0) {
        return BLACK;
    } else if (s < 0) {
        return WHITE;
    }
    return 0;  // Draw
}

void GoBoard::set_from_array(const int8_t* board_data, int8_t to_play) {
    // Copy board data + recompute the Zobrist hash from scratch. PSK
    // history is wiped: we don't have the move sequence that produced
    // this position, so any future PSK check against history would be
    // unsound. Treat this as a clean slate where only the new starting
    // position is "current"; subsequent moves fill seen_hashes_ from
    // here onward.
    current_hash_ = 0;
    for (int i = 0; i < size_ * size_; ++i) {
        board_[i] = board_data[i];
        if (board_[i] != EMPTY) {
            current_hash_ ^= tables_->zobrist[i][board_[i]];
        }
    }
    seen_hashes_.clear();
    to_play_ = to_play;
    ko_point_ = std::nullopt;
    consecutive_passes_ = 0;
    move_count_ = 0;
    for (int i = 0; i < size_ * size_; ++i) {
        if (board_[i] != EMPTY) {
            move_count_++;
        }
    }
}

}  // namespace alpha_go
