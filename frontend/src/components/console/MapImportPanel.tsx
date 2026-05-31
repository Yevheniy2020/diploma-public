import { useRef, useState } from 'react'
import { createMapFromImage, deleteMap } from '../../api/client'
import { useT } from '../../i18n'
import { useAppStore } from '../../state/useAppStore'
import { toast } from '../../state/useToaster'
import type { MapSummary } from '../../types'
import Modal from '../Modal'
import { Section } from './primitives'

function MapImportPanel() {
  const t = useT()
  const currentMap = useAppStore((s) => s.currentMap)
  const maps = useAppStore((s) => s.maps)
  const refreshMaps = useAppStore((s) => s.refreshMaps)
  const loadMap = useAppStore((s) => s.loadMap)

  const [open, setOpen] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [cellSizeM, setCellSizeM] = useState(0.05)
  const [maxCells, setMaxCells] = useState(200)
  const [invert, setInvert] = useState(false)
  const [dilate, setDilate] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [switchingId, setSwitchingId] = useState<number | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<MapSummary | null>(null)
  const [deleting, setDeleting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const onSwitchMap = async (id: number) => {
    if (id === currentMap?.id || switchingId !== null) return
    setSwitchingId(id)
    try {
      await loadMap(id)
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setSwitchingId(null)
    }
  }

  const confirmDelete = async () => {
    if (!deleteTarget) return
    const wasActive = deleteTarget.id === currentMap?.id
    setDeleting(true)
    try {
      await deleteMap(deleteTarget.id)
      const remaining = await refreshMaps()
      if (wasActive && remaining.length > 0) {
        await loadMap(remaining[0].id)
      } else if (wasActive) {
        useAppStore.setState({ currentMap: null, spaces: [], path: [], isMoving: false })
      }
      toast.success(t('map.delete.toast', { name: deleteTarget.name }))
      setDeleteTarget(null)
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setDeleting(false)
    }
  }

  const reset = () => {
    setFile(null)
    setName('')
    setCellSizeM(0.05)
    setMaxCells(200)
    setInvert(false)
    setDilate(0)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const close = () => {
    if (submitting) return
    setOpen(false)
    reset()
  }

  const onSubmit = async () => {
    if (!file) {
      toast.error('Оберіть PNG або JPG')
      return
    }
    const trimmed = name.trim()
    if (!trimmed) {
      toast.error('Введіть назву мапи')
      return
    }
    setSubmitting(true)
    try {
      const res = await createMapFromImage(file, trimmed, {
        cellSizeM,
        maxCells,
        invert,
        dilate,
      })
      await refreshMaps()
      await loadMap(res.id)
      toast.success(`Мапу «${res.name}» імпортовано (${res.width_cells}×${res.height_cells})`)
      setOpen(false)
      reset()
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  const onPickFile = (f: File | null) => {
    setFile(f)
    if (f && !name) {
      const base = f.name.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_-]+/g, '_')
      setName(base.toLowerCase())
    }
  }

  return (
    <>
      <Section
        title="map"
        right={
          <button
            type="button"
            onClick={() => setOpen(true)}
            className="upper-mono text-stone-600 hover:text-stone-900 border hairline-strong px-1.5 py-0.5"
            title="import map from PNG/JPG"
          >
            + import
          </button>
        }
      >
        <div className="pb-2 text-[12.5px] text-stone-700">
          {maps.length === 0 ? (
            <div className="px-3 py-1 text-stone-500 italic">no maps</div>
          ) : (
            <ul className="max-h-[104px] overflow-y-auto overflow-x-hidden scrollbar-slim">
              {maps.map((m) => {
                const active = m.id === currentMap?.id
                const busy = switchingId === m.id
                return (
                  <li
                    key={m.id}
                    className={`group flex items-stretch min-w-0 border-l-2 ${
                      active ? 'border-stone-900 bg-stone-50' : 'border-transparent hover:bg-stone-50'
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => onSwitchMap(m.id)}
                      disabled={busy || (switchingId !== null && !busy)}
                      className={`flex-1 min-w-0 text-left px-3 py-1.5 flex items-baseline justify-between gap-2 ${
                        active ? 'text-stone-900' : 'text-stone-700'
                      } disabled:opacity-50`}
                    >
                      <span className={`truncate min-w-0 ${active ? 'font-medium' : ''}`}>
                        {m.name}
                      </span>
                      <span className="font-mono text-[10.5px] text-stone-400 num shrink-0">
                        {m.width_cells}×{m.height_cells}
                        {busy ? ' · …' : ''}
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        setDeleteTarget(m)
                      }}
                      title={t('map.delete.tooltip')}
                      className="px-2 upper-mono text-red-700 hover:text-red-900 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      ×
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      </Section>

      <Modal open={open} onClose={close} title="import map from image" size="md">
        <div className="space-y-3 text-[13px]">
          <label className="block">
            <span className="upper-mono text-stone-600">PNG / JPG</span>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg"
              onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
              className="block w-full mt-1 text-[12px] file:mr-3 file:py-1 file:px-2 file:border file:hairline-strong file:bg-stone-50 file:text-stone-700 file:upper-mono"
            />
            {file && (
              <span className="block mt-1 font-mono text-[10.5px] text-stone-500 num">
                {file.name} · {(file.size / 1024).toFixed(1)} KB
              </span>
            )}
          </label>

          <label className="block">
            <span className="upper-mono text-stone-600">name</span>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="block w-full mt-1 border hairline px-2 py-1 font-mono"
              placeholder="my_floorplan"
            />
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="upper-mono text-stone-600">cell size · m</span>
              <input
                type="number"
                step="0.01"
                min="0.01"
                max="1"
                value={cellSizeM}
                onChange={(e) => setCellSizeM(Number(e.target.value))}
                className="block w-full mt-1 border hairline px-2 py-1 font-mono num"
              />
            </label>
            <label className="block">
              <span className="upper-mono text-stone-600">max cells</span>
              <input
                type="number"
                step="10"
                min="20"
                max="500"
                value={maxCells}
                onChange={(e) => setMaxCells(Number(e.target.value))}
                className="block w-full mt-1 border hairline px-2 py-1 font-mono num"
              />
            </label>
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-[12px]">
              <input
                type="checkbox"
                checked={invert}
                onChange={(e) => setInvert(e.target.checked)}
              />
              <span className="text-stone-700">invert (light = wall)</span>
            </label>
            <label className="flex items-center gap-2 text-[12px]">
              <span className="text-stone-700">dilate</span>
              <input
                type="number"
                step="1"
                min="0"
                max="5"
                value={dilate}
                onChange={(e) => setDilate(Number(e.target.value))}
                className="w-12 border hairline px-1.5 py-0.5 font-mono num"
              />
              <span className="text-stone-400 text-[10.5px]">px</span>
            </label>
          </div>

          <p className="text-[11px] text-stone-500 italic leading-snug">
            Чорні лінії стануть стінами. Якщо план має білі стіни на темному фоні —
            увімкни «invert». «dilate» зміцнює тонкі лінії після ресайзу.
          </p>

          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={close}
              disabled={submitting}
              className="upper-mono text-stone-600 hover:text-stone-900 px-3 py-1.5 disabled:opacity-50"
            >
              cancel
            </button>
            <button
              type="button"
              onClick={onSubmit}
              disabled={submitting || !file || !name.trim()}
              className="upper-mono bg-stone-900 text-white px-3 py-1.5 disabled:opacity-50"
            >
              {submitting ? 'importing…' : 'import'}
            </button>
          </div>
        </div>
      </Modal>

      <Modal
        open={deleteTarget !== null}
        onClose={() => !deleting && setDeleteTarget(null)}
        title={t('map.delete.title')}
        size="sm"
      >
        <div className="space-y-3 text-[13px]">
          <p>{t('map.delete.confirm', { name: deleteTarget?.name ?? '' })}</p>
          <div className="flex justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={() => setDeleteTarget(null)}
              disabled={deleting}
              className="upper-mono text-stone-600 hover:text-stone-900 px-3 py-1.5 disabled:opacity-50"
            >
              {t('map.delete.cancel')}
            </button>
            <button
              type="button"
              onClick={confirmDelete}
              disabled={deleting}
              className="upper-mono bg-red-700 text-white px-3 py-1.5 disabled:opacity-50"
            >
              {deleting ? t('map.delete.busy') : t('map.delete.button')}
            </button>
          </div>
        </div>
      </Modal>
    </>
  )
}

export default MapImportPanel
