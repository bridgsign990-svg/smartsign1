# BridgeSign project positioning

## Problem
Communications barriers can limit participation in education, emergency response, and daily interaction for Deaf and hard-of-hearing people. In Malawi, accessible communication support must be designed around local needs, inclusion, and practical learning support rather than imported assumptions.

## Proposed solution
BridgeSign is a modular assistive communication platform that combines:

- static letter recognition for supportable handshapes and fingerspelling
- speech-to-text and text-to-speech support for hearing participants
- an avatar-based sign demonstration layer for educational reference
- a learning module structured around guided practice and basic assessment
- emergency communication phrases that can be spoken locally on the device

The platform is explicitly positioned as a research and demonstration prototype for communication-support technology, not as a verified deployed public-service emergency system or a validated MSL recognition engine.

## Target users
- learners practicing communication basics
- Deaf or hard-of-hearing users seeking assistive phrase support
- hearing users who need a simple communication bridge in day-to-day situations
- researchers evaluating a prototype assistive technology pipeline for inclusion-oriented applications

## Research contribution
BridgeSign contributes a practical prototype for:

- assistive sign communication investigation in a low-resource setting
- educational sign-language learning support using structured digital lessons
- multimodal communication bridging between speech, text, and avatar-based sign display
- evaluation-ready architecture for future MSL data collection and validation

## Malawi Pillar 8 alignment
The project aligns most strongly with Pillar 8: Human Capital Development by supporting communication skills, educational access, participation, and inclusive access to information.

### A. Education and skills development
The Learn module offers a lightweight teaching flow: lesson introduction, demonstration, practice, assessment, and progress tracking.

### B. Science, technology and innovation
The project demonstrates computer vision, machine learning, assistive technology, speech processing, and avatar-based interaction in a single prototype platform.

### C. Social welfare and inclusion
BridgeSign addresses communication barriers through accessible features, emergency phrase support, and local communication assistance. It does not claim to replace professional emergency response.

### D. Gender, equity and accessibility
The system is designed for equitable access to communication support, particularly for disabled users and learners, without forcing irrelevant gender-based features.

### E. Human capital development
The design aims to improve communication confidence, independent access to information, and participation in everyday contexts.

### F. Local research capacity
The project is grounded in the need for Malawian Sign Language (MSL) data collection and local validation. The current implementation is intentionally honest that dynamic MSL recognition remains future work.

## Current implementation
Evidence in the codebase shows the following:

- static recognition and camera-based letter support are operational in a controlled prototype setting
- learning content and progress tracking are present and structured for demonstration
- emergency phrase support is implemented as local speech playback only
- avatar and speech modules are active for interaction demonstrations
- dynamic gesture recognition is present as experimental engineering infrastructure, not a validated production feature
- ASL experimental datasets and pipelines are clearly separated from MSL work

## Current evidence
The project can credibly claim:

- a functioning prototype assistive communication interface
- interactive learning content with lesson structure and assessment flow
- proof-of-concept static recognition support for selected sign labels
- experimental dynamic recognition engineering infrastructure with explicit research disclaimers
- local data-collection workflows for future MSL development

## Current limitations
The project must not claim the following as validated status:

- real-time MSL translation at production quality
- emergency alerting or emergency dispatch capability
- generalised dynamic sign recognition for all words or phrases
- a validated MSL-trained model without a reviewed Malawian dataset
- educational efficacy results without participant evaluation

## Future work
Future work should focus on:

- MSL dataset collection and human validation
- local vocabulary development and context-specific phrase sets
- learner evaluation and usability testing
- refinement of dynamic recognition only after curated MSL data becomes available
- stronger evidence linking the system to accessibility and educational outcomes in the Malawi context
