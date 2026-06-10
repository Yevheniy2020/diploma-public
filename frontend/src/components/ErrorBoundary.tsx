import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallbackLabel?: string
}

interface State {
  error: Error | null
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('[ErrorBoundary]', error, info.componentStack)
  }

  reset = (): void => this.setState({ error: null })

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 bg-stone-50 text-stone-900">
        <div className="upper-mono text-[11px] text-red-700">
          {this.props.fallbackLabel ?? 'render error'}
        </div>
        <pre className="text-[12px] font-mono max-w-[80%] max-h-[40%] overflow-auto bg-white border hairline p-3 text-stone-700">
          {error.message}
        </pre>
        <button
          type="button"
          onClick={this.reset}
          className="upper-mono px-3 py-1.5 border-2 border-stone-900 bg-stone-900 text-white hover:bg-stone-800"
        >
          reset
        </button>
      </div>
    )
  }
}
