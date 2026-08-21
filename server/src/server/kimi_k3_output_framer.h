// Kimi K3 output-frame parser.
//
// K3 frame names are ordinary BPE tokens between special <|open|>/<|close|>
// and <|sep|> tokens.  Filtering special tokens independently therefore leaks
// names such as "response" and "message" into API content.  This small,
// request-scoped parser recognizes complete structural frames while preserving
// payload bytes and unknown (for example future tool) frames verbatim.

#pragma once

#include <string>

namespace dflash::common {

struct KimiK3FramedPiece {
    std::string text;
    bool think_boundary = false;
};

class KimiK3OutputFramer {
public:
    KimiK3FramedPiece push(const std::string & raw_token,
                           const std::string & decoded_piece);

    // Flush an incomplete frame verbatim.  Malformed/unknown syntax must not
    // silently hide model output.
    KimiK3FramedPiece finish();

private:
    enum class Marker { None, Open, Close };

    KimiK3FramedPiece complete_frame();
    KimiK3FramedPiece flush_verbatim();

    Marker marker_ = Marker::None;
    std::string frame_text_;
    std::string frame_name_;
};

}  // namespace dflash::common
