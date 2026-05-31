import Modal from './Modal'

interface Props {
  open: boolean
  onClose: () => void
}

function MicPermissionDialog({ open, onClose }: Props) {
  return (
    <Modal open={open} onClose={onClose} title="microphone access blocked">
      <div className="flex flex-col gap-3 text-[13px] text-stone-700">
        <p>
          Voice control needs microphone access. Your browser is currently blocking
          it for this site.
        </p>
        <p>
          In Chrome / Edge: open
          {' '}
          <code className="font-mono text-stone-900">
            chrome://settings/content/microphone
          </code>
          {' '}
          (copy the link), remove this site from the blocked list, then reload. In
          Firefox / Safari: click the lock icon in the address bar and allow
          microphone access.
        </p>
        <div className="flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="upper-mono px-3 py-1.5 border hairline-strong bg-white hover:bg-stone-50 text-stone-700"
          >
            close
          </button>
        </div>
      </div>
    </Modal>
  )
}

export default MicPermissionDialog
