import { useEffect } from 'react'

interface Props {
  open: boolean
  onClose: () => void
  title?: string
  size?: 'sm' | 'md' | 'lg'
  children: React.ReactNode
}

function Modal({ open, onClose, title, size = 'sm', children }: Props) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-stone-900/40"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className={`bg-white border hairline-strong shadow-2xl w-full p-5 ${
          size === 'lg' ? 'max-w-2xl' : size === 'md' ? 'max-w-md' : 'max-w-sm'
        }`}
      >
        {title ? (
          <h3 className="upper-mono text-stone-700 mb-3">{title}</h3>
        ) : null}
        {children}
      </div>
    </div>
  )
}

export default Modal
