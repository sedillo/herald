/**
 * AudioRecorder — thin wrapper around the browser MediaRecorder API.
 *
 * Usage:
 *   const recorder = new AudioRecorder();
 *   await recorder.start();             // requests mic permission, begins recording
 *   const blob = await recorder.stop(); // stops, releases mic, returns Blob
 *   recorder.recording                  // true while active
 */
class AudioRecorder {
  constructor() {
    this._mediaRecorder = null;
    this._chunks = [];
    this._stream = null;
  }

  get recording() {
    return this._mediaRecorder?.state === 'recording';
  }

  async start() {
    this._stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this._chunks = [];

    // Prefer webm/opus; fall back to whatever the browser supports
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : '';

    this._mediaRecorder = new MediaRecorder(
      this._stream,
      mimeType ? { mimeType } : {}
    );

    this._mediaRecorder.addEventListener('dataavailable', e => {
      if (e.data.size > 0) this._chunks.push(e.data);
    });

    // Collect in 250ms slices for granular chunks
    this._mediaRecorder.start(250);
  }

  stop() {
    return new Promise((resolve, reject) => {
      if (!this._mediaRecorder) {
        reject(new Error('Not recording'));
        return;
      }
      this._mediaRecorder.addEventListener('stop', () => {
        const mimeType = this._mediaRecorder.mimeType || 'audio/webm';
        const blob = new Blob(this._chunks, { type: mimeType });
        // Release the mic immediately
        this._stream.getTracks().forEach(t => t.stop());
        this._stream = null;
        this._mediaRecorder = null;
        resolve(blob);
      });
      this._mediaRecorder.stop();
    });
  }
}
