// Run with: node --test mvc/mos/public/i18n.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const C = require('./lt_core.js');
const I = require('./i18n.js');

test('Vietnamese is the default and both languages have the same keys and placeholders', () => {
  assert.equal(I.DEFAULT_LANG, 'vi');
  assert.deepEqual(Object.keys(I.STRINGS.vi).sort(), Object.keys(I.STRINGS.en).sort());
  const holes = (s) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
  for (const k of Object.keys(I.STRINGS.en)) assert.deepEqual(holes(I.STRINGS.vi[k]), holes(I.STRINGS.en[k]), k);
  assert.equal(I.t('xx', 'title'), I.STRINGS.vi.title);
  assert.equal(I.t('en', 'hpProgress', {i: 2, n: 6}), 'Sound check: trial 2 of 6');
  assert.equal(I.t('vi', 'hpProgress', {i: 2, n: 6}), 'Kiểm tra âm thanh: lượt 2 trên 6');
});

test('every key the page uses exists', () => {
  const src = fs.readFileSync(path.join(__dirname, 'lt_app.js'), 'utf8');
  const used = new Set([...src.matchAll(/\bT\('(\w+)'[,)]/g)].map((m) => m[1]));
  for (const k of ['name_mos', 'name_smos', 'name_cmos', 'q_mos', 'q_smos', 'q_cmos', 'tone1', 'tone2', 'tone3',
                   'att_one', 'att_pair', 'hpFailTitle', 'hpFailText', 'finishTitle', 'finishText', 'expired']) {
    used.add(k);
  }
  for (const k of used) assert.ok(k in I.STRINGS.en && k in I.STRINGS.vi, k);
});

test('the scales keep the values of the journal page in both languages', () => {
  const values = {mos: [5, 4, 3, 2, 1], smos: [5, 4, 3, 2, 1], cmos: [3, 2, 1, 0, -1, -2, -3]};
  for (const lang of I.LANGS) {
    for (const [kind, v] of Object.entries(values)) assert.deepEqual(I.scale(lang, kind).map((x) => x[0]), v);
  }
});

test('the background questions are translated, and the stored values stay the English options', () => {
  for (const q of C.BACKGROUND) {
    assert.ok(I.BACKGROUND_VI.questions[q.key], q.key);
    for (const o of q.options.concat([C.NO_ANSWER])) {
      assert.ok(I.BACKGROUND_VI.options[o], o);
      assert.equal(I.bgOption('en', o), o);
    }
  }
  assert.deepEqual(Object.keys(I.BACKGROUND_VI.questions), C.BACKGROUND.map((q) => q.key));
});

// mvc/LISTENING_ADDENDUM_mvc_v1.md §5: the wording, as written there.
test('the scales, questions, instruction and attention texts are the addendum\'s (§5)', () => {
  assert.equal(I.t('en', 'q_mos'), 'How natural does the speech in this recording sound?');
  assert.equal(I.t('vi', 'q_mos'), 'Giọng nói trong đoạn ghi âm này nghe tự nhiên đến mức nào?');
  assert.deepEqual(I.scale('en', 'mos'), [[5, 'Excellent'], [4, 'Good'], [3, 'Fair'], [2, 'Poor'], [1, 'Bad']]);
  assert.deepEqual(I.scale('vi', 'mos'), [[5, 'Xuất sắc'], [4, 'Tốt'], [3, 'Trung bình'], [2, 'Kém'], [1, 'Rất kém']]);
  assert.ok(I.t('en', 'mosIntro2').startsWith('For each recording, rate how natural the speech sounds, that is, how ' +
                                              'much it sounds like a real person speaking. Ignore what is being said.'));
  assert.ok(I.t('vi', 'mosIntro2').startsWith('Với mỗi đoạn, hãy chấm mức độ tự nhiên của giọng nói, tức là nghe giống ' +
                                              'một người thật đang nói đến mức nào. Không để ý đến nội dung được nói.'));
  assert.equal(I.t('en', 'q_smos'), 'How similar is the voice in the Sample to the voice in the Reference? Ignore the ' +
                                    'words and the audio quality.');
  assert.equal(I.t('vi', 'q_smos'), 'Giọng trong đoạn Mẫu giống giọng trong đoạn Tham chiếu đến mức nào? Không để ý ' +
                                    'đến nội dung lời nói và chất lượng âm thanh.');
  assert.deepEqual(I.scale('en', 'smos'), [[5, 'Definitely the same speaker'], [4, 'Probably the same'],
                                           [3, 'Not sure'], [2, 'Probably different'],
                                           [1, 'Definitely different speakers']]);
  assert.deepEqual(I.scale('vi', 'smos'), [[5, 'Chắc chắn cùng một người nói'], [4, 'Có lẽ cùng một người'],
                                           [3, 'Không chắc'], [2, 'Có lẽ là hai người khác nhau'],
                                           [1, 'Chắc chắn là hai người khác nhau']]);
  assert.ok(!JSON.stringify(I.SCALES).includes('khác người'));   // the old label 2
  assert.equal(I.t('en', 'att_one', {r: '2 (Poor)'}), 'To show you are paying attention, please select 2 (Poor) for ' +
                                                      'this sample.');
  assert.equal(I.t('en', 'att_pair', {r: '4 (Probably the same)'}), 'To show you are paying attention, please ' +
                                                                   'select 4 (Probably the same) for this pair.');
  assert.equal(I.t('vi', 'att_one', {r: '2 (Kém)'}), 'Để cho thấy bạn đang chú ý, vui lòng chọn 2 (Kém) cho đoạn ' +
                                                    'ghi âm này.');
  assert.equal(I.t('vi', 'att_pair', {r: '4 (Có lẽ cùng một người)'}), 'Để cho thấy bạn đang chú ý, vui lòng chọn 4 ' +
                                                                     '(Có lẽ cùng một người) cho cặp này.');
});

test('background question 6 and the no-answer option in both languages (§7.4)', () => {
  assert.equal(I.bgQuestion('vi', {key: 'previous_test', question: 'x'}),
               'Bạn đã từng tham gia một bài kiểm tra nghe của nhóm nghiên cứu này chưa?');
  assert.equal(I.bgOption('vi', 'Prefer not to say'), 'Không muốn trả lời');
  assert.equal(C.BACKGROUND.length, 6);
  assert.match(I.t('en', 'bgTitle'), /^Six /);
  assert.match(I.t('vi', 'bgTitle'), /^Sáu /);
});
