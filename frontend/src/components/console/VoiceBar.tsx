import { useState } from 'react'
import VoiceButton from '../VoiceButton'
import { useT } from '../../i18n'
import { useAppStore } from '../../state/useAppStore'
import type { Intent } from '../../types'
import { AudioBars, Section } from './primitives'

interface Props {
  onAudio: (blob: Blob, mime: string) => void | Promise<void>
  processing: boolean
}

function VoiceBar({ onAudio, processing }: Props) {
  const t = useT()
  const lastResp = useAppStore((s) => s.lastVoiceResponse)
  const lastTranscription = useAppStore((s) => s.lastTranscription)
  const recState = useAppStore((s) => s.voicePipelineState)
  const setPendingCorrection = useAppStore((s) => s.setPendingCorrection)
  const [audioLevel, setAudioLevel] = useState(0)

  const canCorrectLast =
    lastResp != null &&
    typeof lastResp.command_log_id === 'number' &&
    lastResp.intent !== 'UNCERTAIN' &&
    lastResp.intent !== 'UNKNOWN'

  const onCorrectLast = () => {
    if (!canCorrectLast || !lastResp) return
    setPendingCorrection({
      command_log_id: lastResp.command_log_id as number,
      transcription: lastTranscription || '',
      predicted_intent: lastResp.intent as Intent,
      predicted_params: lastResp.params ?? {},
      confidence: 1.0,
    })
  }

  const stateLabel: Record<typeof recState, string> = {
    idle: t('voice.state.idle'),
    recording: t('voice.state.recording'),
    uploading: t('voice.state.uploading'),
    parsing: t('voice.state.parsing'),
    done: t('voice.state.done'),
  }

  const transcriptText = lastTranscription

  return (
    <Section title={t('voice.title')}>
      <div className="px-3 pb-3 space-y-3">
        <div className="flex items-center gap-3">
          <VoiceButton onAudio={onAudio} onLevel={setAudioLevel} processing={processing} />
          <div className="flex-1 border hairline-strong bg-[#fafaf7] px-3 py-2 min-h-[64px] flex flex-col justify-center">
            <div className="flex items-center justify-between gap-2">
              <span className="upper-mono text-stone-500 truncate">
                {stateLabel[recState]}
              </span>
              {recState === 'recording' && <AudioBars level={audioLevel} />}
              {(recState === 'parsing' || recState === 'uploading') && (
                <span className="font-mono text-[10.5px] text-stone-500 blink">●●●</span>
              )}
            </div>
            <div className="font-display text-[15px] mt-1 min-h-[20px] text-stone-900">
              {transcriptText ? (
                <>
                  «{' '}
                  <span className="italic">{transcriptText}</span>
                  {' '}»
                </>
              ) : (
                <span className="text-stone-400 italic text-[13px]">{t('voice.transcript.placeholder')}</span>
              )}
            </div>
          </div>
        </div>

        {canCorrectLast ? (
          <button
            type="button"
            onClick={onCorrectLast}
            title={t('voice.correctLast.title')}
            className="upper-mono w-full px-3 py-1.5 border hairline-strong bg-white hover:bg-stone-50 text-stone-700"
          >
            {t('voice.correctLast')}
          </button>
        ) : null}
      </div>
    </Section>
  )
}

export default VoiceBar
