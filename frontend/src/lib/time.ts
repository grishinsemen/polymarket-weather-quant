/**
 * Format a station's current local wall clock and countdown to end of local settlement day.
 */
export function formatStationLocalClock(
  nowIsoOrDate: string | Date,
  localDateIso: string,
  timezone: string,
): { localClock: string; countdown: string } {
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

  const secondsRemaining = dayDelta * 86_400 + (86_400 - currentSecondsOfDay);
  if (secondsRemaining <= 0) {
    return { localClock, countdown: "SETTLED" };
  }

  const hours = Math.floor(secondsRemaining / 3600);
  const mins = Math.floor((secondsRemaining % 3600) / 60);
  return { localClock, countdown: `${hours}h ${String(mins).padStart(2, "0")}m` };
}
