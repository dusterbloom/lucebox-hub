#include "kimi_k3_output_framer.h"

#include <cctype>

namespace dflash::common {

namespace {

constexpr const char * kOpen = "<|open|>";
constexpr const char * kClose = "<|close|>";
constexpr const char * kSep = "<|sep|>";
constexpr size_t kMaxFrameNameBytes = 96;

bool is_special(const std::string & raw) {
    return raw.size() >= 4 && raw.rfind("<|", 0) == 0 &&
           raw.compare(raw.size() - 2, 2, "|>") == 0;
}

bool valid_frame_name_piece(const std::string & text) {
    for (unsigned char c : text) {
        if (std::isalnum(c) || c == '_' || c == '-' || c == ' ' ||
            c == '=' || c == '\"' || c == '\'' || c == '.') {
            continue;
        }
        return false;
    }
    return true;
}

}  // namespace

KimiK3FramedPiece KimiK3OutputFramer::push(
        const std::string & raw_token, const std::string & decoded_piece) {
    if (marker_ == Marker::None) {
        if (raw_token == kOpen || raw_token == kClose) {
            marker_ = raw_token == kOpen ? Marker::Open : Marker::Close;
            frame_text_ = raw_token;
            frame_name_.clear();
            return {};
        }
        return {decoded_piece, false};
    }

    if (raw_token == kSep) {
        frame_text_ += raw_token;
        return complete_frame();
    }

    // A second special token before <|sep|>, an implausible name, or an
    // oversized name is malformed.  Preserve every byte rather than applying
    // the server's generic blanket special-token suppression.
    if (is_special(raw_token) ||
        !valid_frame_name_piece(decoded_piece) ||
        frame_name_.size() + decoded_piece.size() > kMaxFrameNameBytes) {
        frame_text_ += decoded_piece.empty() ? raw_token : decoded_piece;
        return flush_verbatim();
    }

    frame_text_ += decoded_piece;
    frame_name_ += decoded_piece;
    return {};
}

KimiK3FramedPiece KimiK3OutputFramer::complete_frame() {
    KimiK3FramedPiece out;

    // These are the structural frames proven by the K3 GGUF template and
    // committed-token traces.  A normal word "response" in payload text is
    // never touched because it is not bracketed by marker and separator.
    if (frame_name_ == "response" || frame_name_ == "message" ||
        frame_name_ == "message role=\"assistant\"") {
        // Structural envelope only.
    } else if (frame_name_ == "think") {
        out.text = marker_ == Marker::Open ? "<think>" : "</think>";
        out.think_boundary = true;
    } else {
        // Tool and future frame grammars are intentionally not guessed.
        out.text = frame_text_;
    }

    marker_ = Marker::None;
    frame_text_.clear();
    frame_name_.clear();
    return out;
}

KimiK3FramedPiece KimiK3OutputFramer::flush_verbatim() {
    KimiK3FramedPiece out{frame_text_, false};
    marker_ = Marker::None;
    frame_text_.clear();
    frame_name_.clear();
    return out;
}

KimiK3FramedPiece KimiK3OutputFramer::finish() {
    if (marker_ == Marker::None) return {};
    return flush_verbatim();
}

}  // namespace dflash::common
