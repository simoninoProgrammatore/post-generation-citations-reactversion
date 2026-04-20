/**
 * Icon.jsx — Wrapper attorno a lucide-react per usare il naming del mockup.
 *
 * Uso: <Icon name="check" size={14} strokeWidth={2.5} />
 */

import {
  LayoutGrid, Compass, BarChart3, Eye, Microscope,
  BookOpen, Play, Settings, Check, X,
  CheckCircle2, XCircle, Info, AlertTriangle,
  ChevronDown, ChevronRight, ChevronUp,
  Sparkles, Cpu, SlidersHorizontal, Tag, Layers,
  Download, Upload, Search, Filter, RefreshCw,
  Copy, Hash, FileText, Loader, Link2,
} from 'lucide-react'

const REGISTRY = {
  // Nav (stesso naming del mockup)
  pipeline: LayoutGrid,
  explore: Compass,
  metrics: BarChart3,
  attention: Eye,
  interpret: Microscope,

  // Logo/actions
  book: BookOpen,
  play: Play,
  settings: Settings,
  download: Download,
  upload: Upload,

  // Status
  check: Check,
  x: X,
  checkCircle: CheckCircle2,
  xCircle: XCircle,
  info: Info,
  warning: AlertTriangle,

  // Layout
  chevronDown: ChevronDown,
  chevronRight: ChevronRight,
  chevronUp: ChevronUp,

  // Misc
  sparkles: Sparkles,
  cpu: Cpu,
  sliders: SlidersHorizontal,
  tag: Tag,
  layers: Layers,
  link: Link2,
  search: Search,
  filter: Filter,
  refresh: RefreshCw,
  copy: Copy,
  eye: Eye,
  hash: Hash,
  fileText: FileText,
  loader: Loader,
}

export default function Icon({ name, size = 16, strokeWidth = 1.75, color = 'currentColor', style = {} }) {
  const Component = REGISTRY[name]
  if (!Component) {
    console.warn(`Icon "${name}" not found in registry`)
    return null
  }
  return (
    <Component
      size={size}
      strokeWidth={strokeWidth}
      color={color}
      style={{ display: 'inline-block', verticalAlign: 'middle', flexShrink: 0, ...style }}
    />
  )
}