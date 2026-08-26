// Request-scoped parser for Kimi K3's XTML output envelope.
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
