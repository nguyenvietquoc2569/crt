// Fake test data for a visual check of the page: open index.html?sample. The audio files do not exist, so
// require_full_play is off here (the rating buttons would otherwise stay disabled). The real lt_data.js and
// stimuli/ are written by benchmark/listening_stimuli.py.
window.LT_DATA = {
  version: 1,
  config: {
    title: 'Listening test (sample data)',
    consent_html:
      '<p><b>Sample consent text.</b> In the real package this box holds the information sheet: what the study ' +
      'is, how long it takes (about 45 minutes), that taking part is voluntary and you can stop at any time, ' +
      'and that only your rater ID and your answers are stored.</p>' +
      '<p>Please take the test in a quiet room, with headphones.</p>',
    contact: 'organiser@example.org',
    halves: false,
    require_full_play: false,
    headphone_check: {trials: 6, pass: 5, attempts: 2},
    submit_url: null,
  },
  stimuli: {
    '0a1b2c3d': {src: 'stimuli/0a1b2c3d.flac', dur: 2.9},
    '1b2c3d4e': {src: 'stimuli/1b2c3d4e.flac', dur: 3.4},
    '2c3d4e5f': {src: 'stimuli/2c3d4e5f.flac', dur: 2.6},
    '3d4e5f60': {src: 'stimuli/3d4e5f60.flac', dur: 3.1},
    '4e5f6071': {src: 'stimuli/4e5f6071.flac', dur: 4.0},
    '5f607182': {src: 'stimuli/5f607182.flac', dur: 2.8},
    '60718293': {src: 'stimuli/60718293.flac', dur: 3.7},
    '718293a4': {src: 'stimuli/718293a4.flac', dur: 3.3},
    '8293a4b5': {src: 'stimuli/8293a4b5.flac', dur: 5.1},
    '93a4b5c6': {src: 'stimuli/93a4b5c6.flac', dur: 3.0},
    'a4b5c6d7': {src: 'stimuli/a4b5c6d7.flac', dur: 3.5},
    'b5c6d7e8': {src: 'stimuli/b5c6d7e8.flac', dur: 2.7},
    'c6d7e8f9': {src: 'stimuli/c6d7e8f9.flac', dur: 3.9},
    'd7e8f90a': {src: 'stimuli/d7e8f90a.flac', dur: 3.2},
    'e8f90a1b': {src: 'stimuli/e8f90a1b.flac', dur: 3.2},
    'f90a1b2c': {src: 'stimuli/f90a1b2c.flac', dur: 4.4},
    '0a1b2c3e': {src: 'stimuli/0a1b2c3e.flac', dur: 4.4},
    '1b2c3d4f': {src: 'stimuli/1b2c3d4f.flac', dur: 2.5},
    '2c3d4e50': {src: 'stimuli/2c3d4e50.flac', dur: 2.5},
    '3d4e5f61': {src: 'stimuli/3d4e5f61.flac', dur: 5.6},
  },
  practice: [
    {tid: 'prac-1', stim: '0a1b2c3d'},
    {tid: 'prac-2', stim: '1b2c3d4e'},
    {tid: 'prac-3', stim: '2c3d4e5f'},
  ],
  mos: [
    {tid: 'mos-001', stim: '3d4e5f60', half: 'A'},
    {tid: 'mos-002', stim: '4e5f6071', half: 'B'},
    {tid: 'mos-003', stim: '5f607182', half: 'A'},
    {tid: 'mos-004', stim: '60718293', half: 'B'},
    {tid: 'mos-gold-1', stim: '718293a4'},
    {tid: 'mos-att-1', stim: '8293a4b5', attention: 3,
     instruction: 'To show you are paying attention, please select 3 (Fair) for this sample.'},
  ],
  smos: [
    {tid: 'smos-001', stim: '93a4b5c6', ref: 'a4b5c6d7', half: 'A'},
    {tid: 'smos-002', stim: 'b5c6d7e8', ref: 'a4b5c6d7', half: 'B'},
    {tid: 'smos-003', stim: 'c6d7e8f9', ref: 'a4b5c6d7'},
    {tid: 'smos-att-1', stim: 'd7e8f90a', ref: 'a4b5c6d7', attention: 1,
     instruction: 'To show you are paying attention, please select 1 (Definitely different speakers) here.'},
  ],
  cmos: [
    {tid: 'cmos-001', x: 'e8f90a1b', y: 'f90a1b2c'},
    {tid: 'cmos-002', x: '0a1b2c3e', y: '1b2c3d4f'},
    {tid: 'cmos-same-1', x: '2c3d4e50', y: '2c3d4e50'},
    {tid: 'cmos-att-1', x: '3d4e5f61', y: 'f90a1b2c', attention: -2,
     instruction: 'To show you are paying attention, please select −2 (Worse) for this pair.'},
  ],
};
