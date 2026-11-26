/**
 * Format a station's current local wall clock and countdown to end of local settlement day.
 */
export function formatStationLocalClock(
  nowIsoOrDate: string | Date,
  localDateIso: string,
  timezone: string,
): { localClock: string; countdown: string; remainingSeconds: number } {
  const now = typeof nowIsoOrDate === "string" ? new Date(nowIsoOrDate) : nowIsoOrDate;

  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: timezone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(now);

  const lookup = Object.fromEntries(parts.map((p) => [p.type, p.value]));
  const currentLocalDate = `${lookup.year}-${lookup.month}-${lookup.day}`;
  const currentSecondsOfDay =
    Number(lookup.hour) * 3600 + Number(lookup.minute) * 60 + Number(lookup.second);

  const localClock = `${lookup.hour}:${lookup.minute}:${lookup.second}`;

  const [targetYear, targetMonth, targetDay] = localDateIso.split("-").map(Number);
  const [curYear, curMonth, curDay] = currentLocalDate.split("-").map(Number);
  const dayDelta = Math.round(
    (Date.UTC(targetYear, targetMonth - 1, targetDay) -
      Date.UTC(curYear, curMonth - 1, curDay)) /
      86_400_000,
  );

  const secondsRemaining = Math.max(0, dayDelta * 86_400 + (86_400 - currentSecondsOfDay));
  if (secondsRemaining <= 0) {
    return { localClock, countdown: "SETTLED", remainingSeconds: 0 };
  }

  const hours = Math.floor(secondsRemaining / 3600);
  const mins = Math.floor((secondsRemaining % 3600) / 60);
  return {
    localClock,
    countdown: `${hours}h ${String(mins).padStart(2, "0")}m`,
    remainingSeconds: secondsRemaining,
  };
}

/**
 * Format an ISO timestamp or Date as HH:MM (24-hour, no timezone suffix) in the given IANA zone.
 */
export function formatHhMmInZone(isoOrDate: string | Date, timeZone: string): string {
  const date = typeof isoOrDate === "string" ? new Date(isoOrDate) : isoOrDate;
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone,
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(date);
  const lookup = Object.fromEntries(parts.map((p) => [p.type, p.value]));
  const hour = lookup.hour === "24" ? "00" : lookup.hour;
  return `${hour}:${lookup.minute}`;
}

/**
 * Format an ISO timestamp or Date as Moscow time (Europe/Moscow, HH:MM without Z).
 */
export function formatMoscowHhMm(isoOrDate: string | Date): string {
  return formatHhMmInZone(isoOrDate, "Europe/Moscow");
}

/**
 * Format elapsed whole minutes since a Routine METAR observation as "(N мин назад)".
 */
export function formatMetarMinutesAgo(observedAtUtc: string | Date, nowMs: number): string {
  const obsMs =
    typeof observedAtUtc === "string"
      ? new Date(observedAtUtc).getTime()
      : observedAtUtc.getTime();
  const mins = Math.max(0, Math.floor((nowMs - obsMs) / 60_000));
  return `(${mins} мин назад)`;
}

const SHORT_MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

/**
 * Format an ISO timestamp or Date as YYYY-MM-DD in the given IANA timezone.
 */
export function formatIsoDateInZone(isoOrDate: string | Date, timeZone: string): string {
  const date = typeof isoOrDate === "string" ? new Date(isoOrDate) : isoOrDate;
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);
  const lookup = Object.fromEntries(parts.map((p) => [p.type, p.value]));
  return `${lookup.year}-${lookup.month}-${lookup.day}`;
}

/**
 * Format an ISO timestamp or Date as compact "Nov 26 07:18" in the given IANA timezone.
 */
export function formatShortDateTimeInZone(isoOrDate: string | Date, timeZone: string): string {
  const date = typeof isoOrDate === "string" ? new Date(isoOrDate) : isoOrDate;
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone,
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(date);
  const lookup = Object.fromEntries(parts.map((p) => [p.type, p.value]));
  const monthName = SHORT_MONTHS[Number(lookup.month) - 1] ?? lookup.month;
  const day = Number(lookup.day);
  const hour = lookup.hour === "24" ? "00" : lookup.hour;
  return `${monthName} ${day} ${hour}:${lookup.minute}`;
}

/**
 * Format an ISO local settlement date (YYYY-MM-DD) with relative day prefix
 * ("Today", "Tomorrow", "+2d", "Yesterday") against the reference clock in the given timezone.
 */
export function formatScreenerDateLabel(
  localDateIso: string,
  nowMs: number,
  timeZone = "Europe/Moscow",
): string {
  const [targetYear, targetMonth, targetDay] = localDateIso.split("-").map(Number);
  if (!targetYear || !targetMonth || !targetDay) return localDateIso;

  const shortDate = `${SHORT_MONTHS[targetMonth - 1] ?? String(targetMonth)} ${targetDay}`;

  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date(nowMs));
  const lookup = Object.fromEntries(parts.map((p) => [p.type, p.value]));
  const curYear = Number(lookup.year);
  const curMonth = Number(lookup.month);
  const curDay = Number(lookup.day);

  const dayDelta = Math.round(
    (Date.UTC(targetYear, targetMonth - 1, targetDay) -
      Date.UTC(curYear, curMonth - 1, curDay)) /
      86_400_000,
  );

  if (dayDelta === 0) return `Today · ${shortDate}`;
  if (dayDelta === 1) return `Tomorrow · ${shortDate}`;
  if (dayDelta === 2) return `+2d · ${shortDate}`;
  if (dayDelta === -1) return `Yesterday · ${shortDate}`;
  return shortDate;
}
