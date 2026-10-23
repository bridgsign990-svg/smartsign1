window.BRIDGESIGN_PERF = {
  live: {
    useClientHands: true,
    inferIntervalMs: 80,
    frameSkip: 1,
    mediaPipeMaxDim: 320,
    serverFrameMaxDim: 320,
    jpegQuality: 0.55,
    slowWarnMs: 180,
    hardTimeoutMs: 1800,
    localHandsTimeoutMs: 1200,
    staleClearMs: 350,
    noHandFallbackFrames: 90,
    logEvery: 60,
    consecutiveThreshold: 5,
    wordConsecutiveThreshold: 5,
    mediaPipe: {
      maxNumHands: 1,
      modelComplexity: 0,
      minDetectionConfidence: 0.45,
      minTrackingConfidence: 0.45
    }
  },
  speech: {
    requestTimeoutMs: 650,
    interimDebounceMs: 220,
    cacheEntries: 80,
    avatarFallbackMs: 350,
    rate: 1.15
  },
  voice: {
    rate: 1.15,
    pitch: 1.0,
    useLocalTTS: true
  },
  image: {
    maxDim: 512,
    jpegQuality: 0.62,
    timeoutMs: 8000,
    cacheEntries: 32
  },
  call: {
    inferIntervalMs: 85,
    frameSkip: 1,
    mediaPipeMaxDim: 320,
    serverFrameMaxDim: 320,
    jpegQuality: 0.55,
    hardTimeoutMs: 550,
    slowWarnMs: 90,
    speechPartialSendMs: 250,
    maxVideoBitrate: 550000,
    consecutiveThreshold: 3,
    mediaPipe: {
      maxNumHands: 1,
      modelComplexity: 0,
      minDetectionConfidence: 0.45,
      minTrackingConfidence: 0.45
    },
    dataChannel: {
      ordered: false,
      maxRetransmits: 0
    }
  }
};
