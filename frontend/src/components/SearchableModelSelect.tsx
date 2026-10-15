/** Searchable model selector (combobox) for the shared provider catalogue.
 *
 * A compact dropdown-style trigger shows only the selected model; opening it
 * reveals a popup with a search input on top and the filtered model options
 * below (case-insensitive partial match on model ID). Filtering preserves the
 * FREE/PAID optgroup structure via ``groups``; a group with an empty label
 * renders its options flat (Realtime STT). The popup supports keyboard use
 * (ArrowUp/Down to highlight, Enter to select, Escape/Tab to dismiss), closes
 * on selection and on clicks outside, keeps the highlighted option scrolled
 * into view, and shows an explicit no-match message. No new dependencies.
 */

import { useEffect, useRef, useState } from 'react';

interface ModelGroup {
  /** Optgroup label; an empty string renders the models flat. */
  label: string;
  models: string[];
}

interface SearchableModelSelectProps {
  value: string;
  onChange: (value: string) => void;
  groups: ModelGroup[];
  /** Leading option, e.g. the "default" (empty-value) entry. */
  defaultOption?: { value: string; label: string };
  disabled?: boolean;
  /** Style override for the closed trigger (defaults match the shared UI). */
  selectClassName?: string;
  /** Style override for the wrapper (e.g. w-full in grid cells). */
  className?: string;
}

interface FlatOption {
  value: string;
  label: string;
  group: string;
  index: number;
}

const _DEFAULT_TRIGGER_CLS =
  'text-xs bg-gray-800 text-gray-200 border border-gray-700';

export default function SearchableModelSelect({
  value,
  onChange,
  groups,
  defaultOption,
  disabled = false,
  selectClassName = _DEFAULT_TRIGGER_CLS,
  className = 'min-w-0',
}: SearchableModelSelectProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [highlight, setHighlight] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const q = query.trim().toLowerCase();

  const visibleGroups = (
    q
      ? groups.map((g) => ({
          ...g,
          models: g.models.filter((m) => m.toLowerCase().includes(q)),
        }))
      : groups
  ).filter((g) => g.models.length > 0);

  // Flat option list (default entry first) — indexes drive keyboard nav.
  const options: FlatOption[] = [
    ...(defaultOption
      ? [
          {
            value: defaultOption.value,
            label: defaultOption.label,
            group: '',
            index: 0,
          },
        ]
      : []),
    ...visibleGroups.flatMap((g) =>
      g.models.map((m) => ({ value: m, label: m, group: g.label, index: 0 })),
    ),
  ].map((opt, index) => ({ ...opt, index }));

  const matchCount = visibleGroups.reduce((n, g) => n + g.models.length, 0);
  const selectedLabel =
    value === '' ? (defaultOption?.label ?? 'Default') : value;

  const closePopup = (refocus: boolean) => {
    setOpen(false);
    setQuery('');
    if (refocus) triggerRef.current?.focus();
  };

  const openPopup = () => {
    if (disabled) return;
    setQuery('');
    const current = options.findIndex((o) => o.value === value);
    setHighlight(current >= 0 ? current : 0);
    setOpen(true);
  };

  const select = (next: string) => {
    onChange(next);
    closePopup(true);
  };

  // Focus the search input whenever the popup opens.
  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  // Keep the highlighted option visible while navigating.
  useEffect(() => {
    if (!open) return;
    const el = rootRef.current?.querySelector<HTMLElement>(
      `[data-option-index="${highlight}"]`,
    );
    el?.scrollIntoView({ block: 'nearest' });
  }, [highlight, open]);

  // Close on clicks outside the component.
  useEffect(() => {
    if (!open) return;
    const onDocMouseDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        closePopup(false);
      }
    };
    document.addEventListener('mousedown', onDocMouseDown);
    return () => document.removeEventListener('mousedown', onDocMouseDown);
  }, [open]);

  // Group consecutive options under header rows for rendering.
  const segments: { header: string | null; options: FlatOption[] }[] = [];
  options.forEach((opt) => {
    const header = opt.group || null;
    const last = segments[segments.length - 1];
    if (last && last.header === header) {
      last.options.push(opt);
    } else {
      segments.push({ header, options: [opt] });
    }
  });

  return (
    <div ref={rootRef} className={`relative ${className}`}>
      <button
        ref={triggerRef}
        type="button"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => (open ? closePopup(false) : openPopup())}
        onKeyDown={(e) => {
          if (!open && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
            e.preventDefault();
            openPopup();
          }
        }}
        className={`flex w-full items-center justify-between gap-1 rounded px-2 py-1 text-left disabled:opacity-50 ${selectClassName}`}
      >
        <span className="truncate">{selectedLabel}</span>
        <span aria-hidden className="shrink-0 text-gray-400">▾</span>
      </button>

      {open && (
        <div className="absolute left-0 right-0 top-full z-20 mt-1 rounded border border-gray-700 bg-gray-800 shadow-lg">
          <div className="border-b border-gray-700 p-1.5">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setHighlight(0);
              }}
              onKeyDown={(e) => {
                if (e.key === 'ArrowDown') {
                  e.preventDefault();
                  setHighlight((h) => Math.min(h + 1, options.length - 1));
                } else if (e.key === 'ArrowUp') {
                  e.preventDefault();
                  setHighlight((h) => Math.max(h - 1, 0));
                } else if (e.key === 'Enter') {
                  e.preventDefault();
                  const opt = options[highlight];
                  if (opt) select(opt.value);
                } else if (e.key === 'Escape') {
                  e.preventDefault();
                  closePopup(true);
                } else if (e.key === 'Tab') {
                  setOpen(false);
                  setQuery('');
                }
              }}
              placeholder="Search models…"
              className="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1 text-xs text-gray-200 placeholder:text-gray-500 focus:border-blue-600 focus:outline-none"
            />
          </div>
          <div className="max-h-64 overflow-y-auto py-1" role="listbox">
            {segments.map((seg) => (
              <div key={seg.header ?? '__flat__'}>
                {seg.header && (
                  <div className="px-2 pb-0.5 pt-1.5 text-[10px] font-medium uppercase tracking-wide text-gray-500">
                    {seg.header}
                  </div>
                )}
                {seg.options.map((opt) => {
                  const isSelected = opt.value === value;
                  const isHighlighted = opt.index === highlight;
                  return (
                    <button
                      key={`${opt.group}-${opt.value}`}
                      type="button"
                      role="option"
                      aria-selected={isSelected}
                      data-option-index={opt.index}
                      onMouseDown={(e) => e.preventDefault()}
                      onMouseEnter={() => setHighlight(opt.index)}
                      onClick={() => select(opt.value)}
                      className={`flex w-full items-center justify-between gap-2 px-2 py-1 text-left text-xs ${
                        isHighlighted ? 'bg-gray-700 text-white' : 'text-gray-200'
                      } ${isSelected ? 'font-medium text-blue-300' : ''}`}
                    >
                      <span className="truncate">{opt.label}</span>
                      {isSelected && <span aria-hidden>✓</span>}
                    </button>
                  );
                })}
              </div>
            ))}
            {q !== '' && matchCount === 0 && (
              <div className="px-2 py-2 text-xs text-gray-500">
                No models match “{query.trim()}”
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
