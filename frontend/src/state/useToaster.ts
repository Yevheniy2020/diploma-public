import { create } from 'zustand'

export type ToastKind = 'info' | 'success' | 'error'

export interface Toast {
  id: number
  kind: ToastKind
  message: string
}

interface ToasterState {
  toasts: Toast[]
  push: (kind: ToastKind, message: string) => void
  dismiss: (id: number) => void
}

let _seq = 0
const DEFAULT_TTL_MS = 3000

export const useToaster = create<ToasterState>()((set, get) => ({
  toasts: [],
  push: (kind, message) => {
    const id = ++_seq
    set((s) => ({ toasts: [...s.toasts, { id, kind, message }] }))
    setTimeout(() => get().dismiss(id), DEFAULT_TTL_MS)
  },
  dismiss: (id) =>
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}))

export const toast = {
  info: (m: string) => useToaster.getState().push('info', m),
  success: (m: string) => useToaster.getState().push('success', m),
  error: (m: string) => useToaster.getState().push('error', m),
}
