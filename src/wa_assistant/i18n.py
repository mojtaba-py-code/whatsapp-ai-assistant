"""Every customer-facing (and owner-facing) string, in Spanish, Portuguese and English."""

from __future__ import annotations

from datetime import date, datetime

from wa_assistant.text import Lang

MESSAGES: dict[str, dict[Lang, str]] = {
    # --- menu / generic --------------------------------------------------------------------
    "greeting": {
        "es": (
            "¡Hola! Soy el asistente virtual de {business}. Puedo ayudarte con:\n"
            "- Preguntas sobre precios, horarios, ubicación, pagos y más: escribí tu pregunta.\n"
            '- Pedir un turno: escribí "turno".\n'
            '- Hablar con una persona del equipo: escribí "humano".'
        ),
        "pt": (
            "Olá! Sou o assistente virtual da {business}. Posso ajudar com:\n"
            "- Dúvidas sobre preços, horários, endereço, pagamentos e mais: escreva sua pergunta.\n"
            '- Marcar uma consulta: escreva "agendar".\n'
            '- Falar com uma pessoa da equipe: escreva "atendente".'
        ),
        "en": (
            "Hi! I'm the virtual assistant of {business}. I can help you with:\n"
            "- Questions about prices, opening hours, location, payments and more: just ask.\n"
            '- Booking an appointment: type "appointment".\n'
            '- Talking to a person from the team: type "human".'
        ),
    },
    "unsupported_type": {
        "es": "Por ahora solo puedo leer mensajes de texto. ¿Me lo escribís, por favor?",
        "pt": "Por enquanto só consigo ler mensagens de texto. Pode escrever, por favor?",
        "en": "For now I can only read text messages. Could you type it, please?",
    },
    "rate_limited": {
        "es": "Estás enviando muchos mensajes seguidos. Esperá un minuto y volvé a escribir, por favor.",
        "pt": "Você está enviando muitas mensagens seguidas. Aguarde um minuto e escreva de novo, por favor.",
        "en": "You're sending a lot of messages in a row. Please wait a minute and write again.",
    },
    "dont_know": {
        "es": (
            "No tengo esa información en mis documentos y prefiero no inventar. "
            '¿Querés hablar con una persona del equipo? Escribí "humano".'
        ),
        "pt": (
            "Não tenho essa informação nos meus documentos e prefiro não inventar. "
            'Quer falar com uma pessoa da equipe? Escreva "atendente".'
        ),
        "en": (
            "I don't have that information in my documents and I'd rather not guess. "
            'Would you like to talk to a person from the team? Type "human".'
        ),
    },
    "refusal": {
        "es": 'No puedo ayudarte con eso por este medio, pero una persona del equipo sí. Escribí "humano" y te comunico.',
        "pt": 'Não posso ajudar com isso por aqui, mas uma pessoa da equipe pode. Escreva "atendente" que eu transfiro.',
        "en": "I can't help with that here, but a person from the team can. Type \"human\" and I'll connect you.",
    },
    "source": {"es": "Fuente", "pt": "Fonte", "en": "Source"},
    # --- human handoff -----------------------------------------------------------------------
    "handoff_started": {
        "es": "Listo, le aviso a una persona del equipo para que siga la conversación por este chat. Mientras tanto no voy a responder automáticamente.",
        "pt": "Pronto, vou avisar uma pessoa da equipe para continuar a conversa por este chat. Enquanto isso, não vou responder automaticamente.",
        "en": "Done, I'm letting a person from the team know so they can continue this chat. Until then I won't send automatic replies.",
    },
    "handoff_after_hours": {
        "es": "Ahora estamos fuera del horario de atención, así que el equipo te va a responder el próximo día hábil: {next_open}.",
        "pt": "Agora estamos fora do horário de atendimento, então a equipe vai responder no próximo dia útil: {next_open}.",
        "en": "We're outside opening hours right now, so the team will reply on the next business day: {next_open}.",
    },
    "handoff_after_hours_today": {
        "es": "Todavía no abrimos: el equipo te va a responder hoy desde las {time}.",
        "pt": "Ainda não abrimos: a equipe vai responder hoje a partir das {time}.",
        "en": "We're not open yet: the team will reply today from {time}.",
    },
    # --- owner alerts and commands (sent in the business's owner_language) -----------------------
    "owner_handoff_alert": {
        "es": (
            "Pedido de atención humana de +{phone}{name}.\n"
            "Último mensaje: {last}\n"
            "El bot queda en silencio para ese número por {hours} h.\n"
            "Responder: /reply {phone} <mensaje>\nReactivar el bot: /bot on {phone}"
        ),
        "pt": (
            "Pedido de atendimento humano de +{phone}{name}.\n"
            "Última mensagem: {last}\n"
            "O bot fica em silêncio para esse número por {hours} h.\n"
            "Responder: /reply {phone} <mensagem>\nReativar o bot: /bot on {phone}"
        ),
        "en": (
            "Human handoff requested by +{phone}{name}.\n"
            "Last message: {last}\n"
            "The bot stays silent for this number for {hours} h.\n"
            "Reply: /reply {phone} <message>\nTurn the bot back on: /bot on {phone}"
        ),
    },
    "owner_forward": {
        "es": "+{phone} (atención humana) escribió:\n{text}\nResponder: /reply {phone} <mensaje>",
        "pt": "+{phone} (atendimento humano) escreveu:\n{text}\nResponder: /reply {phone} <mensagem>",
        "en": "+{phone} (handed off) wrote:\n{text}\nReply: /reply {phone} <message>",
    },
    "owner_booking_alert": {
        "es": "Nueva solicitud de turno #{id} de +{phone}: {service}, {when}, paciente: {name}. Confirmala con el paciente.",
        "pt": "Nova solicitação de consulta #{id} de +{phone}: {service}, {when}, paciente: {name}. Confirme com o paciente.",
        "en": "New appointment request #{id} from +{phone}: {service}, {when}, patient: {name}. Please confirm it with the patient.",
    },
    "owner_bot_on": {
        "es": "Listo: el bot vuelve a responder a +{phone}.",
        "pt": "Pronto: o bot volta a responder para +{phone}.",
        "en": "Done: the bot is answering +{phone} again.",
    },
    "owner_reply_sent": {
        "es": "Mensaje enviado a +{phone}.",
        "pt": "Mensagem enviada para +{phone}.",
        "en": "Message sent to +{phone}.",
    },
    "owner_reply_failed": {
        "es": "WhatsApp no aceptó el mensaje para +{phone}. Si el cliente no escribió en las últimas 24 h, hace falta una plantilla aprobada.",
        "pt": "O WhatsApp não aceitou a mensagem para +{phone}. Se o cliente não escreveu nas últimas 24 h, é preciso um modelo aprovado.",
        "en": "WhatsApp did not accept the message for +{phone}. If the customer has not written in the last 24 h, an approved template is required.",
    },
    "owner_usage": {
        "es": "Comandos: /bot on <número> (reactiva el bot para ese cliente) | /reply <número> <mensaje> (le responde al cliente).",
        "pt": "Comandos: /bot on <número> (reativa o bot para esse cliente) | /reply <número> <mensagem> (responde ao cliente).",
        "en": "Commands: /bot on <number> (turns the bot back on for that customer) | /reply <number> <message> (replies to the customer).",
    },
    # --- booking flow ---------------------------------------------------------------------------
    "booking_ask_service": {
        "es": '¡Vamos a pedir tu turno! ¿Qué servicio necesitás? Respondé con el número:\n{options}\n(Escribí "cancelar" para salir.)',
        "pt": 'Vamos marcar sua consulta! Qual serviço você precisa? Responda com o número:\n{options}\n(Escreva "cancelar" para sair.)',
        "en": 'Let\'s book your appointment! Which service do you need? Reply with the number:\n{options}\n(Type "cancel" to exit.)',
    },
    "booking_invalid_service": {
        "es": "No reconocí esa opción. Respondé con un número del 1 al {count}:\n{options}",
        "pt": "Não reconheci essa opção. Responda com um número de 1 a {count}:\n{options}",
        "en": "I didn't recognise that option. Reply with a number from 1 to {count}:\n{options}",
    },
    "booking_ask_date": {
        "es": 'Perfecto: {service}. ¿Para qué fecha? Escribí DD/MM (por ejemplo {example}) o "mañana".',
        "pt": 'Perfeito: {service}. Para qual data? Escreva DD/MM (por exemplo {example}) ou "amanhã".',
        "en": 'Great: {service}. What date? Type DD/MM (for example {example}) or "tomorrow".',
    },
    "booking_invalid_date": {
        "es": 'No entendí la fecha. Escribí DD/MM, DD/MM/AAAA o "mañana".',
        "pt": 'Não entendi a data. Escreva DD/MM, DD/MM/AAAA ou "amanhã".',
        "en": 'I couldn\'t read that date. Type DD/MM, DD/MM/YYYY or "tomorrow".',
    },
    "booking_past_date": {
        "es": "Esa fecha ya pasó. ¿Qué otra fecha te queda bien?",
        "pt": "Essa data já passou. Qual outra data fica boa para você?",
        "en": "That date has already passed. Which other date works for you?",
    },
    "booking_too_far": {
        "es": "Solo tomamos solicitudes hasta {days} días adelante (hasta el {last}). ¿Qué fecha más cercana te queda bien?",
        "pt": "Só aceitamos solicitações até {days} dias à frente (até {last}). Qual data mais próxima fica boa?",
        "en": "We only take requests up to {days} days ahead (until {last}). Which earlier date works for you?",
    },
    "booking_closed_day": {
        "es": "El {day} la clínica está cerrada. Atendemos: {hours}. ¿Qué otra fecha te queda bien?",
        "pt": "No dia {day} a clínica está fechada. Atendemos: {hours}. Qual outra data fica boa?",
        "en": "The clinic is closed on {day}. We are open: {hours}. Which other date works for you?",
    },
    "booking_today_over": {
        "es": "Por hoy ya no quedan horarios. ¿Qué otra fecha te queda bien?",
        "pt": "Hoje não há mais horários. Qual outra data fica boa?",
        "en": "There are no slots left today. Which other date works for you?",
    },
    "booking_ask_time": {
        "es": "¿A qué hora? El {day} el primer turno es a las {open} y el último a las {last}. Escribí la hora como HH:MM.",
        "pt": "Que horas? No dia {day} o primeiro horário é às {open} e o último às {last}. Escreva o horário como HH:MM.",
        "en": "What time? On {day} the first slot is at {open} and the last one at {last}. Type the time as HH:MM.",
    },
    "booking_invalid_time": {
        "es": "No entendí la hora. Escribí HH:MM, por ejemplo 10:30.",
        "pt": "Não entendi o horário. Escreva HH:MM, por exemplo 10:30.",
        "en": "I couldn't read that time. Type HH:MM, for example 10:30.",
    },
    "booking_time_closed": {
        "es": "A esa hora no hay turnos. El {day} podés elegir entre las {open} y las {last}.",
        "pt": "Não há horários nesse momento. No dia {day} você pode escolher entre {open} e {last}.",
        "en": "There are no slots at that time. On {day} you can choose between {open} and {last}.",
    },
    "booking_time_past": {
        "es": "Ese horario ya pasó. Elegí uno más tarde, hasta las {last}.",
        "pt": "Esse horário já passou. Escolha um mais tarde, até {last}.",
        "en": "That time has already passed. Choose a later one, up to {last}.",
    },
    "booking_ask_name": {
        "es": "¿A nombre de quién pido el turno? (nombre y apellido del paciente)",
        "pt": "Em nome de quem marco a consulta? (nome e sobrenome do paciente)",
        "en": "Who is the appointment for? (patient's first and last name)",
    },
    "booking_invalid_name": {
        "es": "Escribí el nombre del paciente, por favor (solo letras, hasta 60 caracteres).",
        "pt": "Escreva o nome do paciente, por favor (só letras, até 60 caracteres).",
        "en": "Please type the patient's name (letters only, up to 60 characters).",
    },
    "booking_confirm": {
        "es": 'Revisá tu solicitud:\n- Servicio: {service}\n- Fecha: {when}\n- Paciente: {name}\n¿Confirmás? Respondé "sí" o "no".',
        "pt": 'Confira sua solicitação:\n- Serviço: {service}\n- Data: {when}\n- Paciente: {name}\nConfirma? Responda "sim" ou "não".',
        "en": 'Please check your request:\n- Service: {service}\n- Date: {when}\n- Patient: {name}\nConfirm? Reply "yes" or "no".',
    },
    "booking_confirm_retry": {
        "es": 'Respondé "sí" para confirmar o "no" para descartar la solicitud.',
        "pt": 'Responda "sim" para confirmar ou "não" para descartar a solicitação.',
        "en": 'Reply "yes" to confirm or "no" to discard the request.',
    },
    "booking_saved": {
        "es": "¡Listo! Registramos tu solicitud de turno #{id}. El equipo te va a confirmar el horario por este chat dentro del horario de atención.",
        "pt": "Pronto! Registramos sua solicitação de consulta #{id}. A equipe vai confirmar o horário por este chat dentro do horário de atendimento.",
        "en": "Done! We've recorded your appointment request #{id}. The team will confirm the time in this chat during opening hours.",
    },
    "booking_declined": {
        "es": 'De acuerdo, descarté la solicitud. Si querés empezar de nuevo, escribí "turno".',
        "pt": 'Tudo bem, descartei a solicitação. Se quiser começar de novo, escreva "agendar".',
        "en": 'Alright, I discarded the request. To start again, type "appointment".',
    },
    "booking_cancelled": {
        "es": "Listo, cancelé la solicitud de turno. ¿Te ayudo con algo más?",
        "pt": "Pronto, cancelei a solicitação de consulta. Posso ajudar com mais alguma coisa?",
        "en": "Okay, I cancelled the appointment request. Can I help with anything else?",
    },
}

WEEKDAYS: dict[Lang, tuple[str, ...]] = {
    "es": ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"),
    "pt": (
        "segunda-feira",
        "terça-feira",
        "quarta-feira",
        "quinta-feira",
        "sexta-feira",
        "sábado",
        "domingo",
    ),
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
}

_RANGE_WORD: dict[Lang, str] = {"es": "a", "pt": "a", "en": "to"}
_AT_WORD: dict[Lang, str] = {"es": "a las", "pt": "às", "en": "at"}


def t(key: str, lang: Lang, **values: object) -> str:
    """The ``key`` message in ``lang``, with ``values`` substituted."""
    template = MESSAGES[key][lang]
    return template.format(**values) if values else template


def weekday_name(day: date, lang: Lang) -> str:
    return WEEKDAYS[lang][day.weekday()]


def format_day(day: date, lang: Lang) -> str:
    """``"jueves 15/10/2026"`` / ``"quinta-feira 15/10/2026"`` / ``"Thursday 15/10/2026"``."""
    return f"{weekday_name(day, lang)} {day:%d/%m/%Y}"


def format_when(moment: datetime, lang: Lang) -> str:
    """``"jueves 15/10/2026 a las 10:30"`` (the moment is already in business-local time)."""
    return f"{format_day(moment.date(), lang)} {_AT_WORD[lang]} {moment:%H:%M}"


def day_range(first: int, last: int, lang: Lang) -> str:
    """``"lunes a viernes"`` for weekday indexes 0..4 (a single day when ``first == last``)."""
    names = WEEKDAYS[lang]
    if first == last:
        return names[first]
    return f"{names[first]} {_RANGE_WORD[lang]} {names[last]}"
