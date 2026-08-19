/**
 * A very small declarative control panel.
 *
 * Deliberately dependency-free: the point of this project is what happens on the
 * GPU, and a UI framework would be more code than the renderer.
 */

export interface SliderSpec {
  readonly kind: 'slider';
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  get(): number;
  set(value: number): void;
  /** Formats the current value for display. */
  readonly format?: (v: number) => string;
}

export interface ToggleSpec {
  readonly kind: 'toggle';
  readonly label: string;
  get(): boolean;
  set(value: boolean): void;
}

export interface SelectSpec<T extends string = string> {
  readonly kind: 'select';
  readonly label: string;
  readonly options: readonly T[];
  get(): T;
  set(value: T): void;
}

export interface ButtonSpec {
  readonly kind: 'button';
  readonly label: string;
  onClick(): void;
}

export type ControlSpec = SliderSpec | ToggleSpec | SelectSpec | ButtonSpec;

export interface ControlGroup {
  readonly title: string;
  readonly controls: readonly ControlSpec[];
  readonly collapsed?: boolean;
}

/** Re-reads every control's value into the DOM; call when settings change elsewhere. */
export type Refresh = () => void;

export function buildPanel(host: HTMLElement, groups: readonly ControlGroup[]): Refresh {
  const refreshers: Refresh[] = [];
  host.replaceChildren();

  for (const group of groups) {
    const section = document.createElement('details');
    section.className = 'group';
    section.open = !group.collapsed;
    const summary = document.createElement('summary');
    summary.textContent = group.title;
    section.append(summary);

    for (const control of group.controls) {
      section.append(buildControl(control, refreshers));
    }
    host.append(section);
  }

  const refresh: Refresh = () => {
    for (const r of refreshers) r();
  };
  refresh();
  return refresh;
}

function buildControl(spec: ControlSpec, refreshers: Refresh[]): HTMLElement {
  const row = document.createElement('div');
  row.className = `control control--${spec.kind}`;

  if (spec.kind === 'button') {
    const button = document.createElement('button');
    button.textContent = spec.label;
    button.addEventListener('click', () => spec.onClick());
    row.append(button);
    return row;
  }

  const label = document.createElement('label');
  label.textContent = spec.label;
  row.append(label);

  if (spec.kind === 'slider') {
    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    const readout = document.createElement('span');
    readout.className = 'readout';

    const show = () => {
      const v = spec.get();
      input.value = String(v);
      readout.textContent = spec.format ? spec.format(v) : v.toFixed(2);
    };
    input.addEventListener('input', () => {
      spec.set(Number(input.value));
      show();
    });
    refreshers.push(show);
    row.append(input, readout);
    return row;
  }

  if (spec.kind === 'toggle') {
    const input = document.createElement('input');
    input.type = 'checkbox';
    const show = () => {
      input.checked = spec.get();
    };
    input.addEventListener('change', () => {
      spec.set(input.checked);
      show();
    });
    refreshers.push(show);
    row.append(input);
    return row;
  }

  const select = document.createElement('select');
  for (const option of spec.options) {
    const el = document.createElement('option');
    el.value = option;
    el.textContent = option;
    select.append(el);
  }
  const show = () => {
    select.value = spec.get();
  };
  select.addEventListener('change', () => {
    spec.set(select.value);
    show();
  });
  refreshers.push(show);
  row.append(select);
  return row;
}
