import { useSyncExternalStore } from 'react'

export type Locale = 'en' | 'ua'

const LS_KEY = 'diploma.locale'

const dict = {
  // Header / chrome
  'header.tag': { en: 'UA + EN', ua: 'УКР + АНГ' },
  'header.title': { en: 'Semantic voice navigation', ua: 'Семантична голосова навігація' },
  'header.subtitle': {
    en: '— mobile robot operator console.',
    ua: '— консоль оператора мобільного робота.',
  },
  'meta.backend.ok': { en: 'backend · ok', ua: 'бекенд · ok' },
  'meta.backend.down': { en: 'backend · down', ua: 'бекенд · недоступний' },

  // Mode toggles + toolbar
  'mode.2d': { en: '2D · top-down', ua: '2D · згори' },
  'mode.3d': { en: '3D · perspective', ua: '3D · перспектива' },
  'toggle.grid': { en: 'grid', ua: 'сітка' },
  'toggle.inflation': { en: 'inflation', ua: 'інфляція' },
  'toolbar.hint.3d': { en: 'drag → orbit · scroll → zoom', ua: 'drag → орбіта · scroll → масштаб' },
  'toolbar.hint.2d': { en: 'click on map → set goal', ua: 'клік на мапу → ціль' },

  // Scene placeholders
  'scene.loading': { en: 'loading…', ua: 'завантаження…' },
  'scene.error': { en: 'error', ua: 'помилка' },
  'scene.noMap': { en: 'no map loaded', ua: 'мапу не завантажено' },
  'scene.renderError': { en: 'scene · {mode} render error', ua: 'сцена · помилка рендеру {mode}' },

  // Voice panel
  'voice.title': { en: 'Voice', ua: 'Голос' },
  'voice.state.idle': { en: 'standby · press to speak', ua: 'готовий · натисніть, щоб говорити' },
  'voice.state.recording': { en: '◉ recording · opus webm', ua: '◉ запис · opus webm' },
  'voice.state.uploading': { en: '↑ uploading to /api/voice', ua: '↑ відправка на /api/voice' },
  'voice.state.parsing': { en: '◌ llama-4 · parsing', ua: '◌ llama-4 · розбір' },
  'voice.state.done': { en: '✓ intent dispatched', ua: '✓ команду виконано' },
  'voice.transcript.placeholder': { en: '— transcription —', ua: '— транскрипція —' },
  'voice.correctLast': { en: 'correct last command', ua: 'виправити останню команду' },
  'voice.correctLast.title': {
    en: 'open correction dialog for the last command',
    ua: 'відкрити діалог виправлення останньої команди',
  },
  'voice.error': { en: 'voice: {msg}', ua: 'голос: {msg}' },

  // Labels panel
  'labels.title': { en: 'Semantic labels', ua: 'Семантичні мітки' },
  'labels.add': { en: '+ add', ua: '+ додати' },
  'labels.empty': {
    en: '— no labels yet. Use voice or "+ add" at robot pose.',
    ua: '— міток ще немає. Використайте голос або "+ додати" біля робота.',
  },
  'labels.edit': { en: 'edit', ua: 'редагувати' },
  'labels.delete': { en: 'delete', ua: 'видалити' },
  'labels.toast.pathNotFound': { en: 'path not found: {name}', ua: 'шлях не знайдено: {name}' },
  'labels.toast.going': { en: 'going to {name}', ua: 'прямую до {name}' },
  'labels.toast.deleted': { en: 'deleted: {name}', ua: 'видалено: {name}' },

  // Add-label dialog
  'addLabel.title': { en: 'add label at robot', ua: 'нова мітка на позиції робота' },
  'addLabel.name': { en: 'name', ua: 'назва' },
  'addLabel.radius': { en: 'radius (m)', ua: 'радіус (м)' },
  'addLabel.description': {
    en: 'description (optional, helps voice match)',
    ua: 'опис (необов’язково, допомагає голосу)',
  },
  'addLabel.descriptionPlaceholder': { en: 'e.g. where I cook', ua: 'напр. де я готую' },
  'addLabel.position': { en: 'position', ua: 'позиція' },
  'addLabel.create': { en: 'create', ua: 'створити' },
  'addLabel.error.name': { en: 'name is required', ua: 'назва обовʼязкова' },
  'addLabel.error.radius': {
    en: 'radius must be between 0.2 and 5.0',
    ua: 'радіус має бути від 0.2 до 5.0',
  },
  'addLabel.toast.created': { en: 'created: {name}', ua: 'створено: {name}' },

  // Label editor (inline)
  'labelEditor.descriptionPlaceholder': {
    en: 'description (where I eat)',
    ua: 'опис (напр. де я їм)',
  },

  // Training panel
  'training.title': { en: 'Training', ua: 'Навчання' },
  'training.pending': { en: 'pending corrections', ua: 'корекції в очікуванні' },
  'training.lastRetrained': { en: 'last retrained', ua: 'останнє тренування' },
  'training.lastRetrained.today': { en: 'today {hhmm}', ua: 'сьогодні {hhmm}' },
  'training.crashed': { en: 'training crashed', ua: 'тренування зірвалось' },
  'training.button.running': { en: 'training…', ua: 'тренування…' },
  'training.button.retrain': { en: 'retrain model', ua: 'перетренувати модель' },
  'training.noNew': { en: 'no new corrections', ua: 'нових корекцій немає' },
  'training.bgHint': {
    en: 'you can keep speaking — training runs in the background',
    ua: 'можна продовжувати голос — тренування у фоні',
  },
  'training.title.review': { en: 'review / edit', ua: 'переглянути / редагувати' },
  'training.title.history': { en: 'correction history', ua: 'історія корекцій' },
  'training.toast.completed': {
    en: 'model updated · weights reloaded',
    ua: 'модель оновлено · ваги перезавантажено',
  },
  'training.toast.failed': { en: 'training failed: {err}', ua: 'тренування зірвалось: {err}' },
  'training.toast.started': { en: 'training started · {eta}', ua: 'тренування запущено · {eta}' },
  'training.eta.few': { en: '~3-4 min', ua: '~3-4 хв' },
  'training.eta.many': { en: '~10 min', ua: '~10 хв' },
  'training.phase.preparing.label': { en: 'collecting corrections', ua: 'збираю виправлення' },
  'training.phase.preparing.hint': {
    en: 'reading DB, building seed file',
    ua: 'читаю БД, формую seed-файл',
  },
  'training.phase.preparing.eta': { en: '~1 sec', ua: '~1 сек' },
  'training.phase.paraphrasing.label': { en: 'expanding dataset', ua: 'розширюю датасет' },
  'training.phase.paraphrasing.hint': {
    en: 'Groq generating paraphrases for new examples',
    ua: 'Groq генерує paraphrases для нових прикладів',
  },
  'training.phase.paraphrasing.eta': { en: '~30-60 sec', ua: '~30-60 сек' },
  'training.phase.training.label': { en: 'retraining model', ua: 'перетреную модель' },
  'training.phase.training.hint': {
    en: 'fine-tune XLM-RoBERTa · 8 epochs on MPS',
    ua: 'fine-tune XLM-RoBERTa · 8 епох на MPS',
  },
  'training.phase.training.eta': { en: '~3 min', ua: '~3 хв' },
  'training.phase.done.label': { en: 'done', ua: 'завершено' },
  'training.phase.done.hint': { en: 'new weights loaded', ua: 'нові ваги перезавантажено' },
  'training.phase.failed.label': { en: 'error', ua: 'помилка' },

  // Correction dialog
  'correction.title': { en: 'not quite sure', ua: 'не зовсім впевнений' },
  'correction.transcription': { en: 'transcription', ua: 'транскрипція' },
  'correction.intent': { en: 'intent', ua: 'намір' },
  'correction.confidence': { en: 'confidence', ua: 'впевненість' },
  'correction.cancel': { en: 'no, cancel', ua: 'ні, скасувати' },
  'correction.confirm': { en: 'yes, run', ua: 'так, виконати' },
  'correction.saving': { en: 'saving…', ua: 'збереження…' },
  'correction.toast.saved': { en: 'correction saved', ua: 'виправлення записано' },

  // Corrections list dialog
  'list.title': { en: 'pending corrections', ua: 'корекції що очікують' },
  'list.intent': { en: 'intent', ua: 'намір' },
  'list.confirmDelete': { en: 'really delete?', ua: 'справді видалити?' },
  'list.no': { en: 'no', ua: 'ні' },
  'list.yesDelete': { en: 'yes, delete', ua: 'так, видалити' },
  'list.loading': { en: 'loading…', ua: 'завантаження…' },
  'list.empty': {
    en: 'no corrections pending retraining',
    ua: 'немає корекцій що очікують ретренінгу',
  },
  'list.entry': { en: 'entry', ua: 'запис' },
  'list.entries': { en: 'entries', ua: 'записів' },
  'list.pendingSuffix': { en: 'pending next retrain', ua: 'очікують наступного ретренінгу' },
  'list.close': { en: 'close', ua: 'закрити' },
  'list.edit': { en: '✎ edit', ua: '✎ редагувати' },
  'list.delete': { en: '✕ delete', ua: '✕ видалити' },
  'list.cancel': { en: 'cancel', ua: 'скасувати' },
  'list.save': { en: 'save', ua: 'зберегти' },
  'list.saving': { en: 'saving…', ua: 'збереження…' },
  'list.chip.rejected': { en: 'rejected', ua: 'відхилено' },
  'list.chip.override': { en: 'override', ua: 'перевизначено' },
  'list.chip.slotEdit': { en: 'slot edit', ua: 'правка слотів' },
  'list.modelPrefix': { en: 'model:', ua: 'модель:' },
  'list.toast.updated': { en: 'correction updated', ua: 'корекцію оновлено' },
  'list.toast.deleted': { en: 'correction deleted', ua: 'корекцію видалено' },

  // Slot editor
  'slot.parameters': { en: 'parameters', ua: 'параметри' },
  'slot.choose': { en: '— choose —', ua: '— оберіть —' },

  // Goal panel
  'goal.title': { en: 'Goal · current', ua: 'Ціль · поточна' },
  'goal.x': { en: 'x', ua: 'x' },
  'goal.y': { en: 'y', ua: 'y' },
  'goal.distance': { en: 'distance', ua: 'дистанція' },
  'goal.waypoints': { en: 'waypoints', ua: 'точки шляху' },
  'goal.eta': { en: 'ETA', ua: 'ETA' },
  'goal.stop': { en: '■ STOP', ua: '■ СТОП' },
  'goal.empty': {
    en: '— no active goal. Issue a NAVIGATE command or click the map.',
    ua: '— активної цілі немає. Скажіть NAVIGATE або клікніть на мапу.',
  },

  // Lang switch button
  'lang.toggle': { en: 'language', ua: 'мова' },

  // Real-robot indicator (TopMetaBar)
  'robot.online': { en: 'robot · online', ua: 'робот · онлайн' },
  'robot.offline': { en: 'robot · offline', ua: 'робот · офлайн' },
  'robot.stale': { en: 'robot · stale', ua: 'робот · застаріло' },

  // Map edit
  'edit.title': { en: 'edit', ua: 'редагувати' },
  'edit.off': { en: 'off', ua: 'вимк.' },
  'edit.paint': { en: 'wall+', ua: 'стіна+' },
  'edit.erase': { en: 'wall−', ua: 'стіна−' },
  'edit.hint': {
    en: 'click or drag a cell to add/erase walls',
    ua: 'клік або драг по клітинці щоб додати/стерти стіну',
  },
  'edit.toast.saveFail': {
    en: 'failed to save map: {msg}',
    ua: 'не вдалось зберегти мапу: {msg}',
  },

  // Map delete
  'map.delete.title': { en: 'delete map', ua: 'видалити мапу' },
  'map.delete.tooltip': { en: 'delete map', ua: 'видалити мапу' },
  'map.delete.confirm': {
    en: 'Delete map «{name}»? All its spaces and command history will be erased with it.',
    ua: 'Видалити мапу «{name}»? Усі її простори та історія команд буде стерто разом із нею.',
  },
  'map.delete.cancel': { en: 'cancel', ua: 'скасувати' },
  'map.delete.button': { en: 'delete', ua: 'видалити' },
  'map.delete.busy': { en: 'deleting…', ua: 'видалення…' },
  'map.delete.toast': { en: 'Map «{name}» deleted', ua: 'Мапу «{name}» видалено' },
} as const

export type I18nKey = keyof typeof dict

function readLocale(): Locale {
  if (typeof window === 'undefined') return 'en'
  const v = window.localStorage.getItem(LS_KEY)
  return v === 'ua' ? 'ua' : 'en'
}

let currentLocale: Locale = readLocale()
const listeners = new Set<() => void>()

function emit() {
  for (const l of listeners) l()
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn)
  return () => {
    listeners.delete(fn)
  }
}

export function getLocale(): Locale {
  return currentLocale
}

export function setLocale(next: Locale): void {
  if (next === currentLocale) return
  currentLocale = next
  if (typeof window !== 'undefined') window.localStorage.setItem(LS_KEY, next)
  emit()
}

function interpolate(s: string, params?: Record<string, string | number>): string {
  if (!params) return s
  return s.replace(/\{(\w+)\}/g, (_, k) => (k in params ? String(params[k]) : `{${k}}`))
}

export function t(key: I18nKey, params?: Record<string, string | number>): string {
  return interpolate(dict[key][currentLocale], params)
}

export function useLocale(): Locale {
  return useSyncExternalStore(subscribe, getLocale, getLocale)
}

export function useT(): (key: I18nKey, params?: Record<string, string | number>) => string {
  const locale = useLocale()
  return (key, params) => interpolate(dict[key][locale], params)
}
