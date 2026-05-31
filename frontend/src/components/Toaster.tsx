import { useToaster, type ToastKind } from '../state/useToaster'

const styles: Record<ToastKind, string> = {
  info: 'bg-white border-stone-300 text-stone-900',
  success: 'bg-emerald-700 border-emerald-700 text-white',
  error: 'bg-red-700 border-red-700 text-white',
}

function Toaster() {
  const toasts = useToaster((s) => s.toasts)
  const dismiss = useToaster((s) => s.dismiss)
  if (toasts.length === 0) return null
  return (
    <div className="fixed top-12 right-4 z-50 flex flex-col gap-2 pointer-events-none">
      {toasts.map((t) => (
        <button
          key={t.id}
          onClick={() => dismiss(t.id)}
          className={`pointer-events-auto border px-4 py-2 text-sm shadow-md max-w-xs text-left font-mono ${styles[t.kind]}`}
        >
          {t.message}
        </button>
      ))}
    </div>
  )
}

export default Toaster
