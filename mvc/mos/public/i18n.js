/* Listening test: Vietnamese and English texts of the rater page. Vietnamese is the default. The answers keep the
   English values of lt_core.js (scale numbers, BACKGROUND options), so the results file does not depend on the
   language. In the browser it is window.MOSI18n; in Node, module.exports. */
(function (root) {
  'use strict';

  const LANGS = ['vi', 'en'];
  const DEFAULT_LANG = 'vi';
  const LANG_KEY = 'mos:lang';

  const STRINGS = {
    en: {
      langName: 'English',
      title: 'Listening test',
      loading: 'Loading the listening test…',
      testData: 'TEST DATA: this is a demonstration with synthetic sounds. The answers are not used.',
      notOpen: 'The listening test is not open yet. Please try again later or contact {contact}.',
      cannotStart: 'The test cannot start',
      serverDown: 'The server cannot be reached. Please check your internet connection and reload the page.',
      serverError: 'Something went wrong ({msg}). Please reload the page; your saved answers are kept.',
      organiser: 'the organiser',
      footer: 'Questions or problems: {contact}',
      logout: 'Use another code',
      consentTitle: 'Information and consent',
      agree: 'I agree',
      continue: 'Continue',
      codeTitle: 'Your access code',
      codeText: 'Please enter your access code. Your answers are saved on the server after every response, so you ' +
                'can stop and continue later, on this or another device, with the same code.',
      codePlaceholder: 'e.g. 7K3M9QX2TA4B',
      codeHint: 'The code has 12 letters and digits, sometimes after a short prefix and "-".',
      codeInvalid: 'This code is not valid. Please check it and try again.',
      tooMany: 'Too many attempts. Please wait a few minutes and try again.',
      regClosed: 'New sessions cannot be started now. If you have a code, please enter it above.',
      expired: 'Your session has ended. Please enter your code again to continue.',
      startNew: 'I have no code: start a new session',
      newCodeTitle: 'Your code',
      newCodeText: 'Please write down this code now: it is shown only this once. You need it to continue later, on ' +
                   'this or another device, and to withdraw your answers.',
      welcomeBack: 'Welcome back. Your saved answers were loaded ({n} of {total} answers). If this is not your ' +
                   'code, press "Use another code".',
      pending: 'Your last answers have not reached the server yet. They are kept in this browser and are sent ' +
               'again automatically. Please keep this page open.',
      resynced: 'The page was brought up to date with the answers saved on the server.',
      consentChanged: 'The information text has been updated. Please read it again and confirm that you agree.',
      syncFailed: 'The page could not stay in step with the server. Please reload the page; your saved answers ' +
                  'are kept. If this happens again, please contact {contact}.',
      hpTitle: 'Sound check',
      hpFailedOnce: 'The sound check was not passed. Please make sure that you are wearing headphones with both ' +
                    'sides on your ears, and that the sound does not come from the computer\'s loudspeakers. ' +
                    'Then try once more (attempt {n} of {max}).',
      hpIntro: 'This test must be done with headphones (wired or wireless), not loudspeakers, in a quiet place. ' +
               'It starts with a short check of your headphones.',
      hpStep1: 'Put on your headphones.',
      hpStep2: 'Press "Play test sound" and set the volume of your computer to a comfortable level. You can play ' +
               'it as often as you like.',
      hpStep3: 'In each of the {n} trials you will hear three tones. Choose the one that is quietest. Each trial ' +
               'can be played only once.',
      playTest: 'Play test sound',
      hpStart: 'Start the sound check',
      tryAgain: 'Try again',
      hpProgress: 'Sound check: trial {i} of {n}',
      hpQuestion: 'Which of the three tones was the quietest?',
      hpOnce: 'You can play the tones only once.',
      tone1: 'First tone',
      tone2: 'Second tone',
      tone3: 'Third tone',
      play: 'Play',
      playing: 'Playing…',
      played: 'Played',
      next: 'Next',
      hpPassed: 'Sound check passed. Thank you.',
      hpFailTitle: 'Thank you for your time',
      hpFailText: 'This test has to be done over headphones, and the sound check was not passed, so the test ' +
                  'ends here. This usually depends on the equipment or the sound settings, not on you. Your ' +
                  'participation has been recorded; there is nothing else to do.',
      audioCtxError: 'This browser cannot play the sound-check tones ({msg}). Please use a current version of ' +
                     'Chrome, Firefox or Safari, or contact {contact}.',
      name_mos: 'Naturalness',
      name_smos: 'Speaker similarity',
      name_cmos: 'Comparison',
      partLabel: 'Part {i} of {n}: {name}',
      practice: 'Practice',
      q_mos: 'How natural does the speech in this recording sound?',
      q_smos: 'How similar is the voice in the Sample to the voice in the Reference? Ignore the words and the ' +
              'audio quality.',
      q_cmos: 'Compared with A, the quality of B is:',
      clipPlay: 'Play',
      clipRef: 'Reference',
      clipSample: 'Sample',
      stop: 'Stop',
      playedBadge: 'played',
      hintOne: 'Play the recording to the end to enable the rating.',
      hintBoth: 'Play both recordings to the end to enable the rating.',
      trialProgress: '{head}: trial {i} of {n}',
      overall: 'Overall: {p}% done',
      audioError: 'A recording could not be played. Please reload the page (your answers so far are saved). If ' +
                  'the problem continues, please contact {contact}.',
      att_one: 'To show you are paying attention, please select {r} for this sample.',
      att_pair: 'To show you are paying attention, please select {r} for this pair.',
      fullOne: 'The rating buttons become active once the recording has played to the end.',
      fullBoth: 'The rating buttons become active once both have played to the end.',
      fullAdvice: 'Please listen to each recording to the end before you rate it.',
      mosIntro1: 'You will hear short recordings of speech, one at a time. Some are recordings of real people; ' +
                 'others were produced by computer programs.',
      mosIntro2: 'For each recording, rate how natural the speech sounds, that is, how much it sounds like a real ' +
                 'person speaking. Ignore what is being said. {full} You may play a recording again before you ' +
                 'rate it.',
      practiceIntro: 'We start with {n} practice recordings, so that you can get used to the scale. They do not ' +
                     'count.',
      startPractice: 'Start practice',
      start: 'Start',
      mosAfterPractice: 'The practice is finished. The test itself starts now: {n} recordings, rated in the same ' +
                        'way.',
      mosCount: 'There are {n} recordings.',
      breakText: 'This part is complete. Take a short break if you like, or close the page now and continue later ' +
                 'with the same access code. The test continues when you press Start.',
      midBreakTitle: 'Break',
      midBreakText: 'You are halfway through this part. Take a short break if you like, or close the page now and ' +
                    'continue later with the same access code. The test continues when you press Continue.',
      smosIntro: 'In each trial you hear two recordings: a Reference and a Sample. Rate how similar the voice in ' +
                 'the Sample is to the voice in the Reference: could it be the same person speaking? Ignore the ' +
                 'words and the audio quality.',
      cmosIntro: 'In each trial you hear two versions, A and B, of the same sentence. Compare the overall quality ' +
                 'of B with the quality of A, from +3 (B is much better) to −3 (B is much worse). If you hear no ' +
                 'difference, choose 0 (About the same).',
      playBoth: 'Play both recordings.',
      trialsCount: 'There are {n} trials.',
      bgTitle: 'Six short questions',
      bgText: 'All your ratings are saved. These questions only help us describe the group of listeners in the ' +
              'paper. You can answer any of them with "{no}".',
      finishTitle: 'Thank you!',
      finishText: 'You have completed the listening test and your answers are saved. Thank you for your help.',
      finishPending: 'Your last answers are still being sent. Please keep this page open until this message ' +
                     'disappears. If it stays, download your answers and contact {contact} for help: only answers ' +
                     'that reach the server are used.',
      download: 'Download my answers',
      savedAs: 'Saved as {name} (usually in your Downloads folder).',
    },
    vi: {
      langName: 'Tiếng Việt',
      title: 'Bài kiểm tra nghe',
      loading: 'Đang tải bài kiểm tra nghe…',
      testData: 'TEST DATA (DỮ LIỆU THỬ): đây là bản minh họa với âm thanh tổng hợp. Các câu trả lời không được ' +
                'sử dụng.',
      notOpen: 'Bài kiểm tra nghe chưa mở. Vui lòng quay lại sau hoặc liên hệ {contact}.',
      cannotStart: 'Không thể bắt đầu bài kiểm tra',
      serverDown: 'Không kết nối được với máy chủ. Vui lòng kiểm tra kết nối Internet và tải lại trang.',
      serverError: 'Đã xảy ra lỗi ({msg}). Vui lòng tải lại trang; các câu trả lời đã lưu vẫn được giữ.',
      organiser: 'người tổ chức',
      footer: 'Câu hỏi hoặc sự cố: {contact}',
      logout: 'Dùng mã khác',
      consentTitle: 'Thông tin và đồng ý tham gia',
      agree: 'Tôi đồng ý',
      continue: 'Tiếp tục',
      codeTitle: 'Mã tham gia của bạn',
      codeText: 'Vui lòng nhập mã tham gia của bạn. Câu trả lời của bạn được lưu trên máy chủ sau mỗi lần trả lời, ' +
                'vì vậy bạn có thể dừng lại và làm tiếp sau, trên thiết bị này hoặc thiết bị khác, với cùng mã này.',
      codePlaceholder: 'ví dụ 7K3M9QX2TA4B',
      codeHint: 'Mã gồm 12 chữ cái và chữ số, đôi khi đứng sau một tiền tố ngắn và dấu "-".',
      codeInvalid: 'Mã này không hợp lệ. Vui lòng kiểm tra và thử lại.',
      tooMany: 'Bạn đã thử quá nhiều lần. Vui lòng đợi vài phút rồi thử lại.',
      regClosed: 'Hiện không thể bắt đầu phiên mới. Nếu bạn đã có mã, vui lòng nhập mã ở trên.',
      expired: 'Phiên làm bài đã kết thúc. Vui lòng nhập lại mã để làm tiếp.',
      startNew: 'Tôi chưa có mã: bắt đầu phiên mới',
      newCodeTitle: 'Mã của bạn',
      newCodeText: 'Vui lòng ghi lại mã này ngay: mã chỉ được hiển thị một lần này. Bạn cần mã này để làm tiếp sau, ' +
                   'trên thiết bị này hoặc thiết bị khác, và để rút lại câu trả lời của mình.',
      welcomeBack: 'Chào mừng bạn quay lại. Các câu trả lời đã lưu đã được tải ({n} trên {total} câu). Nếu đây ' +
                   'không phải mã của bạn, hãy bấm "Dùng mã khác".',
      pending: 'Các câu trả lời gần nhất chưa được gửi tới máy chủ. Chúng được giữ trong trình duyệt này và sẽ ' +
               'được tự động gửi lại. Vui lòng không đóng trang.',
      resynced: 'Trang đã được cập nhật theo các câu trả lời đã lưu trên máy chủ.',
      consentChanged: 'Nội dung thông tin đã được cập nhật. Vui lòng đọc lại và xác nhận rằng bạn đồng ý.',
      syncFailed: 'Trang không đồng bộ được với máy chủ. Vui lòng tải lại trang; các câu trả lời đã lưu vẫn được ' +
                  'giữ. Nếu lỗi này lặp lại, vui lòng liên hệ {contact}.',
      hpTitle: 'Kiểm tra âm thanh',
      hpFailedOnce: 'Bạn chưa vượt qua bước kiểm tra âm thanh. Vui lòng kiểm tra rằng bạn đang đeo tai nghe với ' +
                    'cả hai bên áp vào tai, và âm thanh không phát ra từ loa của máy tính. Sau đó thử lại một lần ' +
                    'nữa (lần {n} trên {max}).',
      hpIntro: 'Bài kiểm tra này phải được làm bằng tai nghe (có dây hoặc không dây), không dùng loa, ở nơi yên ' +
               'tĩnh. Bài bắt đầu bằng một bước kiểm tra ngắn tai nghe của bạn.',
      hpStep1: 'Đeo tai nghe vào.',
      hpStep2: 'Nhấn "Phát âm thanh thử" và chỉnh âm lượng máy tính ở mức dễ chịu. Bạn có thể phát lại bao nhiêu ' +
               'lần tùy ý.',
      hpStep3: 'Trong mỗi lượt (tổng cộng {n} lượt), bạn sẽ nghe ba âm. Hãy chọn âm nhỏ nhất. Mỗi lượt chỉ được ' +
               'phát một lần.',
      playTest: 'Phát âm thanh thử',
      hpStart: 'Bắt đầu kiểm tra âm thanh',
      tryAgain: 'Thử lại',
      hpProgress: 'Kiểm tra âm thanh: lượt {i} trên {n}',
      hpQuestion: 'Âm nào trong ba âm là nhỏ nhất?',
      hpOnce: 'Bạn chỉ được phát các âm này một lần.',
      tone1: 'Âm thứ nhất',
      tone2: 'Âm thứ hai',
      tone3: 'Âm thứ ba',
      play: 'Phát',
      playing: 'Đang phát…',
      played: 'Đã phát',
      next: 'Tiếp',
      hpPassed: 'Bạn đã vượt qua bước kiểm tra âm thanh. Cảm ơn bạn.',
      hpFailTitle: 'Cảm ơn bạn đã dành thời gian',
      hpFailText: 'Bài kiểm tra này cần được làm bằng tai nghe, và bước kiểm tra âm thanh chưa đạt, nên bài kiểm ' +
                  'tra kết thúc tại đây. Điều này thường do thiết bị hoặc cài đặt âm thanh, không phải do bạn. Sự ' +
                  'tham gia của bạn đã được ghi nhận; bạn không cần làm gì thêm.',
      audioCtxError: 'Trình duyệt này không phát được các âm kiểm tra ({msg}). Vui lòng dùng phiên bản mới của ' +
                     'Chrome, Firefox hoặc Safari, hoặc liên hệ {contact}.',
      name_mos: 'Độ tự nhiên',
      name_smos: 'Độ giống giọng',
      name_cmos: 'So sánh',
      partLabel: 'Phần {i} trên {n}: {name}',
      practice: 'Luyện tập',
      q_mos: 'Giọng nói trong đoạn ghi âm này nghe tự nhiên đến mức nào?',
      q_smos: 'Giọng trong đoạn Mẫu giống giọng trong đoạn Tham chiếu đến mức nào? Không để ý đến nội dung lời ' +
              'nói và chất lượng âm thanh.',
      q_cmos: 'So với A, chất lượng của B:',
      clipPlay: 'Phát',
      clipRef: 'Tham chiếu',
      clipSample: 'Mẫu',
      stop: 'Dừng',
      playedBadge: 'đã nghe',
      hintOne: 'Hãy nghe hết đoạn ghi âm để có thể chấm điểm.',
      hintBoth: 'Hãy nghe hết cả hai đoạn ghi âm để có thể chấm điểm.',
      trialProgress: '{head}: lượt {i} trên {n}',
      overall: 'Tổng cộng: đã xong {p}%',
      audioError: 'Không phát được một đoạn ghi âm. Vui lòng tải lại trang (các câu trả lời đến giờ đã được lưu). ' +
                  'Nếu vẫn gặp sự cố, vui lòng liên hệ {contact}.',
      att_one: 'Để cho thấy bạn đang chú ý, vui lòng chọn {r} cho đoạn ghi âm này.',
      att_pair: 'Để cho thấy bạn đang chú ý, vui lòng chọn {r} cho cặp này.',
      fullOne: 'Các nút chấm điểm chỉ hoạt động sau khi đoạn ghi âm đã phát hết.',
      fullBoth: 'Các nút chấm điểm chỉ hoạt động sau khi cả hai đoạn đã phát hết.',
      fullAdvice: 'Vui lòng nghe hết mỗi đoạn ghi âm trước khi chấm điểm.',
      mosIntro1: 'Bạn sẽ nghe lần lượt từng đoạn ghi âm giọng nói ngắn. Một số là giọng của người thật; số khác ' +
                 'được tạo ra bởi chương trình máy tính.',
      mosIntro2: 'Với mỗi đoạn, hãy chấm mức độ tự nhiên của giọng nói, tức là nghe giống một người thật đang nói ' +
                 'đến mức nào. Không để ý đến nội dung được nói. {full} Bạn có thể nghe lại một đoạn trước khi ' +
                 'chấm điểm.',
      practiceIntro: 'Chúng ta bắt đầu với {n} đoạn luyện tập để bạn làm quen với thang điểm. Các đoạn này không ' +
                     'được tính.',
      startPractice: 'Bắt đầu luyện tập',
      start: 'Bắt đầu',
      mosAfterPractice: 'Phần luyện tập đã xong. Bài kiểm tra chính thức bắt đầu: {n} đoạn ghi âm, chấm điểm theo ' +
                        'cách tương tự.',
      mosCount: 'Có {n} đoạn ghi âm.',
      breakText: 'Phần này đã hoàn thành. Bạn có thể nghỉ một chút, hoặc đóng trang bây giờ và làm tiếp sau bằng ' +
                 'cùng mã tham gia. Bài kiểm tra sẽ tiếp tục khi bạn nhấn Bắt đầu.',
      midBreakTitle: 'Nghỉ giải lao',
      midBreakText: 'Bạn đã làm được một nửa phần này. Bạn có thể nghỉ một chút, hoặc đóng trang bây giờ và làm ' +
                    'tiếp sau bằng cùng mã tham gia. Bài kiểm tra sẽ tiếp tục khi bạn nhấn Tiếp tục.',
      smosIntro: 'Trong mỗi lượt, bạn nghe hai đoạn ghi âm: Tham chiếu và Mẫu. Hãy chấm mức độ giống nhau giữa ' +
                 'giọng trong đoạn Mẫu và giọng trong đoạn Tham chiếu: đó có thể là cùng một người đang nói không? ' +
                 'Không để ý đến nội dung lời nói và chất lượng âm thanh.',
      cmosIntro: 'Trong mỗi lượt, bạn nghe hai phiên bản A và B của cùng một câu. Hãy so sánh chất lượng tổng thể ' +
                 'của B với A, từ +3 (B tốt hơn nhiều) đến −3 (B kém hơn nhiều). Nếu không nghe thấy khác biệt, ' +
                 'hãy chọn 0 (Như nhau).',
      playBoth: 'Hãy nghe cả hai đoạn.',
      trialsCount: 'Có {n} lượt.',
      bgTitle: 'Sáu câu hỏi ngắn',
      bgText: 'Tất cả điểm đánh giá của bạn đã được lưu. Các câu hỏi này chỉ giúp chúng tôi mô tả nhóm người nghe ' +
              'trong bài báo. Bạn có thể trả lời bất kỳ câu nào bằng "{no}".',
      finishTitle: 'Cảm ơn bạn!',
      finishText: 'Bạn đã hoàn thành bài kiểm tra nghe và các câu trả lời đã được lưu. Cảm ơn sự giúp đỡ của bạn.',
      finishPending: 'Các câu trả lời cuối cùng vẫn đang được gửi. Vui lòng giữ trang mở cho đến khi thông báo này ' +
                     'biến mất. Nếu thông báo vẫn còn, hãy tải câu trả lời xuống và liên hệ {contact} để được hỗ ' +
                     'trợ: chỉ những câu trả lời đã đến máy chủ mới được sử dụng.',
      download: 'Tải câu trả lời của tôi',
      savedAs: 'Đã lưu thành {name} (thường nằm trong thư mục Downloads).',
    },
  };

  // The rating scales by language: [value, label]. The value is what is stored.
  const SCALES = {
    en: {
      mos: [[5, 'Excellent'], [4, 'Good'], [3, 'Fair'], [2, 'Poor'], [1, 'Bad']],
      smos: [[5, 'Definitely the same speaker'], [4, 'Probably the same'], [3, 'Not sure'],
             [2, 'Probably different'], [1, 'Definitely different speakers']],
      cmos: [[3, 'Much better'], [2, 'Better'], [1, 'Slightly better'], [0, 'About the same'],
             [-1, 'Slightly worse'], [-2, 'Worse'], [-3, 'Much worse']],
    },
    vi: {
      mos: [[5, 'Xuất sắc'], [4, 'Tốt'], [3, 'Trung bình'], [2, 'Kém'], [1, 'Rất kém']],
      smos: [[5, 'Chắc chắn cùng một người nói'], [4, 'Có lẽ cùng một người'], [3, 'Không chắc'],
             [2, 'Có lẽ là hai người khác nhau'], [1, 'Chắc chắn là hai người khác nhau']],
      cmos: [[3, 'Tốt hơn nhiều'], [2, 'Tốt hơn'], [1, 'Tốt hơn một chút'], [0, 'Như nhau'],
             [-1, 'Kém hơn một chút'], [-2, 'Kém hơn'], [-3, 'Kém hơn nhiều']],
    },
  };

  // Vietnamese texts of the background questions of lt_core.js (BACKGROUND, NO_ANSWER); English uses them as they are.
  const BACKGROUND_VI = {
    questions: {
      age: 'Nhóm tuổi của bạn',
      first_language: 'Ngôn ngữ mẹ đẻ của bạn',
      english: 'Bạn hiểu tiếng Anh nói ở mức nào?',
      hearing: 'Bạn có vấn đề về thính giác đã biết không?',
      device: 'Bạn đã nghe bằng thiết bị gì?',
      previous_test: 'Bạn đã từng tham gia một bài kiểm tra nghe của nhóm nghiên cứu này chưa?',
    },
    options: {
      '18–24': '18–24', '25–34': '25–34', '35–44': '35–44', '45 or older': '45 trở lên',
      'Vietnamese': 'Tiếng Việt', 'English': 'Tiếng Anh', 'Other': 'Khác',
      'Basic': 'Cơ bản', 'Intermediate': 'Trung cấp', 'Advanced': 'Nâng cao',
      'Native or near-native': 'Như người bản ngữ hoặc gần như vậy',
      'No': 'Không', 'Yes': 'Có', 'Not sure': 'Không chắc',
      'Over-ear or on-ear headphones': 'Tai nghe chụp tai (trùm tai hoặc áp tai)',
      'In-ear earphones or earbuds': 'Tai nghe nhét tai',
      'Prefer not to say': 'Không muốn trả lời',
    },
  };

  function norm(lang) {
    return LANGS.includes(lang) ? lang : DEFAULT_LANG;
  }

  // The text of key in lang with {name} placeholders filled from vars; a missing key falls back to English.
  function t(lang, key, vars) {
    const table = STRINGS[norm(lang)];
    let s = Object.prototype.hasOwnProperty.call(table, key) ? table[key] : STRINGS.en[key];
    if (s === undefined) return key;
    for (const [k, v] of Object.entries(vars || {})) s = s.split('{' + k + '}').join(String(v));
    return s;
  }

  function scale(lang, kind) {
    return SCALES[norm(lang)][kind];
  }

  function bgQuestion(lang, q) {
    return norm(lang) === 'vi' ? BACKGROUND_VI.questions[q.key] || q.question : q.question;
  }

  function bgOption(lang, option) {
    return norm(lang) === 'vi' ? BACKGROUND_VI.options[option] || option : option;
  }

  function initialLang() {
    try {
      return norm(root.localStorage && root.localStorage.getItem(LANG_KEY));
    } catch (e) {
      return DEFAULT_LANG;
    }
  }

  function saveLang(lang) {
    try { root.localStorage.setItem(LANG_KEY, norm(lang)); } catch (e) { /* the default is used next time */ }
  }

  const api = {LANGS, DEFAULT_LANG, STRINGS, SCALES, BACKGROUND_VI, t, scale, bgQuestion, bgOption, initialLang,
               saveLang};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MOSI18n = api;
})(typeof window !== 'undefined' ? window : this);
