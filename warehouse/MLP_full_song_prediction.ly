\paper { 
  indent = 0\mm
}

\header{
  title = "나비야"
  composer = "MLP full song prediction"
}

melody = \relative c'' {
\clef treble
\key c \major
\autoBeamOff
\time 2/4
g8 e8 e4 f8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4 e8 e8 e4
}

\score {
  \new Staff \melody
  \layout { }
  \midi { }
}